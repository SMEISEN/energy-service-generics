"""
The generic parts of the API component of services.

Copyright 2024 FZI Research Center for Information Technology

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

SPDX-FileCopyrightText: 2024 FZI Research Center for Information Technology
SPDX-License-Identifier: Apache-2.0
"""

import os
import sys
import logging
from uuid import UUID

from celery import states
from celery.result import AsyncResult
from fastapi import Depends
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi import status
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.responses import Response
from fastapi.security.open_id_connect_url import OpenIdConnect
from pydantic import ValidationError

# To prevent tests from failing if only parts of the package are used.
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    import uvicorn
except ModuleNotFoundError:
    Instrumentator = None
    uvicorn = None

from esg.models.task import HTTPError
from esg.models.task import HTTPValidationError
from esg.models.task import TaskId
from esg.models.task import TaskStatus
from esg.service.exceptions import GenericUnexpectedException


# Map the built in states of celery to the state definitions of service
# framework. The internal states of celery are documented here:
# https://docs.celeryq.dev/en/stable/userguide/tasks.html#task-states
# NOTE: There is an additional `states.REVOKED` status which is not
#       considered here as the service framework has no functionality
#       to cancel tasks. This might change in future.
# NOTE: Celery matches an unknown ID to the pending state. PENDING is
#       hence not included here.
TASK_STATUS_MAP = {
    states.STARTED: "running",
    states.SUCCESS: "ready",
    states.FAILURE: "ready",
    states.RETRY: "queued",
}


class API:
    """
    API component for services that implements the `/request/` endpoints.

    Attributes:
    -----------
    post_request_responses : dict
        Allows defining additional responses for the POST /request/ endpoint.
        This information is solely used for extending the API schema.
    get_request_status_responses : dict
        Like above but for GET /request/{request_ID}/status/
    get_request_result_responses : dict
        Like above but for GET /request/{request_ID}/result/
    post_request_description : string
        Allows defining the description text for the POST /request/ endpoint.
        This information is solely used for extending the API schema.
    get_request_status_description : dict
        Like above but for GET /request/{request_ID}/status/
    get_request_result_description : dict
        Like above but for GET /request/{request_ID}/result/

    """

    # Manually add the validation error back to responses. FastAPI usually
    # adds this response automatically but does not so if the endpoint
    # uses `Response` as argument instead of models, which is the case for
    # the post methods of the API.
    post_request_responses = {
        422: {
            "model": HTTPValidationError,
            "description": "Validation Error.",
        }
    }

    # This endpoint should only fail if the ID is unknown.
    get_request_status_responses = {
        404: {
            "model": HTTPError,
            "description": "No task with the provided ID exists.",
        }
    }

    # Besides the 404 it could also happen that an error during processing
    # the request occurred. See the docstrings of `GenericUnexpectedException`
    # and `RequestInducedException` for details.
    get_request_result_responses = {
        # NOTE: This should not happen, it means that the input validation
        #       has failed. If you want to reenable this add a test that checks
        #       that `RequestInducedException` thrown in a task is forwarded
        #       to the API and raised there again.
        # 400: {
        #     "model": HTTPError,
        #     "description": (
        #         "Returned if processing the task yields an error that "
        #         "is related to the request arguments. The detail string "
        #         "provides additional information on the error source."
        #     ),
        # },
        404: get_request_status_responses[404],
        409: {
            "model": HTTPError,
            "description": ("The task is not ready yet."),
        },
        500: {
            "model": HTTPError,
            "description": (
                "Processing the task has caused some unexpected "
                "error. Please contact the provider of the service for "
                "support."
            ),
        },
    }

    # Here the description texts for the three endpoints. FastAPI uses
    # the docstrings by default. However, these contain a lot if internal
    # stuff that is not relevant for the user. Hence the here the option
    # to set these explicitly.
    post_request_description = (
        "Create a request task (for e.g. a forecast or optimized schedule) "
        "that is computed in the background."
    )
    get_request_status_description = "Return the status of a request task."
    get_request_result_description = "Return the result of a request task."

    def __init__(
        self,
        RequestInput,
        RequestOutput,
        request_task,
        title,
        description,
        version,
        version_root_path=None,
        fastapi_kwargs={},
    ):
        """
        Init basic stuff like the logger and configure the REST API.

        Configuration is partly taken from arguments and partly from
        environment variables. Here anything that is likely be set in the
        source code of the derived service is expected as argument. Any
        configuration that users want to change for single instances of
        the services are environment variables, e.g. the log level or
        credentials.

        TODO: Add JWT Checking as global dependency to app. See here:
              https://github.com/tiangolo/fastapi/blob/ab22b795903bb9a782ccfc3d2b4e450b565f6bfc/fastapi/applications.py#L332
              https://pyjwt.readthedocs.io/en/latest/usage.html

        Environment variables:
        ----------------------
        LOGLEVEL : str
            The loglevel to use for *all* loggers. Defaults to logging.INFO.
        ROOT_PATH : str
            Allows serving the service under a subpath, e.g.
            `"https://example.com/service-1/v1/"` would require setting
            `ROOT_PATH` to `"/service-1/v1/"`. Note that by convention we
            require the last element of the root path to contain the major
            part of the version number.

        Arguments:
        ----------
        RequestInput : pydantic model
            A Model defining the structure and documentation of the input data,
            i.e. the data that is necessary to process a request.
        RequestOutput : pydantic model
            A Model defining the structure and documentation of the output data,
            i.e. the result of the request.
        request_task : Celery task
            This is the function that is called to process calls to the
            POST /request/ endpoint. Note that this function must be wrapped
            by Celery's task decorator. See here for details:
            https://docs.celeryq.dev/en/stable/userguide/tasks.html
        title : str
            The title (aka name) of the service. Forwarded to FastAPI, see also:
            https://fastapi.tiangolo.com/tutorial/metadata/
        description : str
            The description of the service. Forwarded to FastAPI, see also:
            https://fastapi.tiangolo.com/tutorial/metadata/
        version : `packaging.version.Version` instance
            The version of the service. Is used to extend the schema
            documentation.
        version_root_path : str or None
            Can be used to specify the version name expected in root path.
            If `None` or empty will default to `f"v{version.major}"`.
        fastapi_kwargs : dict
            Additional keyword arguments passed to FastAPI(), May be useful
            to extend schema docs.
        """
        if Instrumentator is None or uvicorn is None:
            raise ModuleNotFoundError(
                "Running a service requires `uvicorn` and "
                "`prometheus_fastapi_instrumentator` to work. If you are "
                "using docker consider using a tag with `-service`."
            )

        self.RequestInput = RequestInput
        self.RequestOutput = RequestOutput
        self.request_task = request_task

        # Log everything to stdout by default, i.e. to docker container logs.
        # This comes on top as we want to emit our initial log message as soon
        # as possible to support debugging if anything goes wrong.
        self._logger_name = "service"
        self.logger = logging.getLogger(self._logger_name)

        # Assumes that a logger that has a handler assigned to it does not
        # need another one additionally. This prevents that multiple copies
        # of log messages are displayed while running tests.
        if not self.logger.handlers:
            stream_handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                "%(levelname)-10s%(asctime)s - %(message)s"
            )
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)
        self.logger.setLevel(logging.INFO)

        self.logger.info("Initiating API.")

        # Parse the requested log level from environment variables and
        # configure all loggers accordingly.
        self._loglevel = getattr(logging, (os.getenv("LOGLEVEL") or "INFO"))
        self.logger.info(
            "Changing log level for all loggers to %s", self._loglevel
        )
        for logger_name in logging.root.manager.loggerDict:
            logging.getLogger(logger_name).setLevel(self._loglevel)

        # Load and check root path for service.
        fastapi_root_path = os.getenv("ROOT_PATH") or f"/v{version.major}"
        if version_root_path is None:
            version_root_path = f"v{version.major}"
        if fastapi_root_path[0] != "/" or fastapi_root_path[-1] == "/":
            raise ValueError(
                "`ROOT_PATH` must have a leading and no trailing slash. "
                f"Got instead: {fastapi_root_path}"
            )
        if fastapi_root_path.split("/")[-1] != version_root_path:
            raise ValueError(
                "`ROOT_PATH` must contain version number at last path element. "
                f"Got instead: {fastapi_root_path}"
            )

        # TODO: Rework this, just a tester yet.
        # The FastAPI docs are not very complete here. This was found here:
        # https://github.com/HarryMWinters/fastapi-oidc/issues/1
        # self.fastapi_oauth2 = OpenIdConnect(
        #     openIdConnectUrl="<the well known URL>",
        #     scheme_name="My Authentication Method",
        # )

        # Define the REST API Endpoint.
        self.fastapi_app = FastAPI(
            title=title,
            description=description,
            docs_url="/",
            redoc_url=None,
            root_path=fastapi_root_path,
            version=str(version),
            dependencies=[
                Depends(self.dummy_dependency),
                # Depends(self.fastapi_oauth2),
            ],
        )
        self.fastapi_app.post(
            "/request/",
            status_code=status.HTTP_201_CREATED,
            response_model=TaskId,
            responses=self.post_request_responses,
            description=self.post_request_description,
            openapi_extra={
                # Allows us to add the schema of `RequestInput` to the API
                # docs, even though we don't use the standard way of patching
                # the model to the endpoint.
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": RequestInput.model_json_schema()
                        }
                    },
                    "required": True,
                },
            },
        )(self.post_request)
        self.fastapi_app.get(
            "/request/{task_id}/status/",
            response_model=TaskStatus,
            responses=self.get_request_status_responses,
            description=self.get_request_status_description,
        )(self.get_request_status)
        self.fastapi_app.get(
            "/request/{task_id}/result/",
            response_model=RequestOutput,
            responses=self.get_request_result_responses,
            description=self.get_request_result_description,
        )(self.get_request_result)

    def dummy_dependency(self):
        print(f"Test: {self.fastapi_app.title}")

    async def post_request(self, request: Request):
        """
        Handle post calls to the request endpoint.

        This checks that the user data matches the related input data model
        and creates a task that is computed by the worker.
        """
        # Validate that the sent data matches the input schema...
        raw_body = await request.body()
        try:
            _ = self.RequestInput.model_validate_json(raw_body)
        except ValidationError as exc:
            # This does the same thing FastAPI does on ValidationErrors. See:
            # https://github.com/tiangolo/fastapi/blob/master/docs_src/handling_errors/tutorial006.py
            return await request_validation_exception_handler(request, exc)

        # ... and forward the JSON data to the worker.
        # Here we use the JSON originally sent by the client to prevent
        # that we need to serialize the parsed data again.
        task = self.request_task.delay(input_data_json=raw_body)

        return TaskId(task_ID=task.id)

    async def get_request_status(self, task_id: UUID):
        """
        This method triggers the computation of the status response, it thus
        answers any calls to the  GET /request/{task_ID}/status/ endpoint.

        Arguments:
        ----------
        task_id : UUID
            As returned by POST /request/

        Returns:
        --------
        task_status : esg.models.task.TaskStatus instance
            The status info of the task.

        Raises:
        -------
        fastapi.HTTPException
            If the task does not exist.

        TODO: You might want to make sure here that a task created
              but not started yet might emit a state that matches the
              `"queued"` state. However, this seems to be a
              celery config variable that does this.
        """
        task = AsyncResult(str(task_id))
        task_state = task.state

        if task_state == states.PENDING:
            error_msg = "Could not find task with ID: %s"
            self.logger.info(error_msg, task_id)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_msg % task_id,
            )

        self.logger.debug(f"Celery state of task is: {task_state}")
        task_status_text = TASK_STATUS_MAP[task_state]

        task_status = TaskStatus(status_text=task_status_text)
        return task_status

    async def get_request_result(self, task_id: UUID):
        """
        This method fetches the result of a task, it thus answers any calls
        to the  GET /request/{task_ID}/result/ endpoint.

        Arguments:
        ----------
        task_id : UUID
            As returned by POST /request/

        Returns:
        --------
        response : fastapi.responses.JSONResponse instance
            The validated output data in a JSON response object.

        Raises:
        -------
        fastapi.HTTPException
            If the task does not exist or is not ready yet.
        esg.services.base.GenericUnexpectedException
            Upon unexpected errors during handling the request,
            see Exception docstring for details.
        esg.services.base.RequestInducedException
            Upon request induced errors during handling the request,
            see Exception docstring for details.
        """
        task = AsyncResult(str(task_id))
        task_state = task.state

        if task_state == states.PENDING:
            error_msg = "Could not find task with ID: %s"
            self.logger.info(error_msg, task_id)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_msg % task_id,
            )
        elif task_state in states.UNREADY_STATES:
            error_msg = "Task is not ready yet. ID was: %s"
            self.logger.info(error_msg, task_id)
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=error_msg % task_id,
            )
        elif task_state == states.FAILURE:
            self.logger.info("Failed task encountered for ID: %s", task_id)
            raise GenericUnexpectedException()

        # Fetch and verify that the output data matches the API docs.
        output_data_json = task.get()
        try:
            self.RequestOutput.model_validate_json(output_data_json)
        except ValidationError:
            self.logger.error(
                "Task computed data not matching the Output data model."
            )
            raise GenericUnexpectedException()

        # Directly return the JSON data. This saves a few CPU
        # cycles by preventing that the data is serialized again.
        response = Response(
            content=output_data_json, media_type="application/json"
        )
        return response

    def close(self):
        """
        Place here anything that needs to be done to clean up.

        That is nothing for the default case of the `API` class as essentials
        parts for graceful shutdown are integrated in the `run` method.
        """
        pass

    def run(self):
        """
        Run the FastAPI app with uvicorn.
        """
        self.logger.info("Initiating API execution.")

        try:
            # Exposes a Prometheus endpoint for monitoring.
            # Instrumentator().instrument(self.fastapi_app).expose(
            #     self.fastapi_app, include_in_schema=False
            # )

            # Serve the REST endpoint.
            # NOTE: uvicorn handles SIGTERM signals for us, that is, this line
            #       below will block until SIGTERM or SIGINT is received.
            #       Afterwards the finally statement is executed.
            uvicorn.run(
                self.fastapi_app,
                host="0.0.0.0",
                port=8800,
            )
        except Exception:
            self.logger.exception(
                "Exception encountered while serving the API."
            )
            raise
        finally:
            # This should be called on system exit and keyboard interrupt.
            self.logger.info("Shutting down API.")
            self.close()
            self.logger.info("API shutdown completed. Good bye!")