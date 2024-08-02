"""
Tests for content of fooc.py

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

from esg.test.generic_tests import GenericFOOCTest
from esg.service.worker import compute_request_input_model
from esg.service.worker import compute_fit_parameters_input_model
import pandas as pd
import pytest


from fooc import AutoPVdefault, fetch_meteo_data
from fooc import handle_request, fit_parameters
from data_model import RequestArguments
from data_model import FittedParameters
from data_model import RequestOutput
from data_model import FitParameterArguments
from data_model import Observations
from .data import OPEN_METEO_RESPONSE_DF
from .data import REQUEST_INPUTS_FOOC_TEST
from .data import REQUEST_OUTPUTS_FOOC_TEST
from .data import FIT_PARAM_INPUTS_FOOC_TEST
from .data import FIT_PARAM_OUTPUTS_FOOC_TEST


def test_predict_pv_power():
    """
    Check that the AutoPV model generally works and produces the expected output.
    """
    expected_power = (
        pd.Series(
            index=[
                pd.Timestamp("2017-04-01 12:00:00-0700", tz="US/Arizona"),
                pd.Timestamp("2017-04-01 13:00:00-0700", tz="US/Arizona"),
            ],
            # Has been computed with checked inputs by AutoPV.
            data=[0.6584965313615996, 0.6791242559472552],
        )
    )
    meteo_data = pd.DataFrame(
        [[1050, 20, 2], [1050, 20, 2]],
        columns=["ghi", "t2m", "ws10"],
        index=[
            pd.Timestamp("20170401 1200", tz="US/Arizona"),
            pd.Timestamp("20170401 1300", tz="US/Arizona"),
        ],
    )
    actual_power = AutoPVdefault(
        latitude=49.01365,
        longitude=8.40444,
        altitude=75.3,
        weights=[1/12]*12,
        peak_power=15.
    ).predict(X=meteo_data)
    # Allow some slight deviations as pvlib updates seem to affect the
    # computed numbers a bit.
    pd.testing.assert_series_equal(actual_power, expected_power, rtol=0.005)


def test_fetch_meteo_data(open_meteo_httpserver):
    """
    Verify that the `fetch_meteo_data` can parse the data in the open
    meteo format to a dataframe matching the needs of `AutoPVdefault.predict()`.
    """
    open_meteo_httpserver.add_request(
        lat=49.01365,
        lon=8.40444,
        past_days=2,
    )

    meteo_data = fetch_meteo_data(lat=49.01365, lon=8.40444, past_days=2)
    pd.testing.assert_frame_equal(meteo_data, OPEN_METEO_RESPONSE_DF)

    # Furthermore verify that the output of `fetch_meteo_data` is compatible
    # with the format expected by `AutoPVdefault.predict()`. This needs no `assert`,
    # it's fine if no exception is raised.
    _ = AutoPVdefault(
        latitude=49.01365,
        longitude=8.40444,
        altitude=75.3,
        weights=[1/12]*12,
        peak_power=15.
    ).predict(X=meteo_data)


RequestInput = compute_request_input_model(
    RequestArguments=RequestArguments,
    FittedParameters=FittedParameters,
)


class TestHandleRequest(GenericFOOCTest):
    InputDataModel = RequestInput
    OutputDataModel = RequestOutput
    input_data_jsonable = [m["JSONable"] for m in REQUEST_INPUTS_FOOC_TEST]
    output_data_jsonable = [m["JSONable"] for m in REQUEST_OUTPUTS_FOOC_TEST]

    def get_payload_function(self):
        return handle_request

    @pytest.fixture(autouse=True)
    def open_meteo_httpserver_request(self, open_meteo_httpserver):
        """
        Enable fake Open Meteo endpoint providing data if query parameters
        match the ones expected by request, i.e. requests data for
        the future only.
        """
        open_meteo_httpserver.add_request(
            # NOTE: lat/lon must match items in `REQUEST_INPUTS_FOOC_TEST`
            lat=49.01365,
            lon=8.40444,
            past_days=0,
        )


FitParametersInput = compute_fit_parameters_input_model(
    FitParameterArguments=FitParameterArguments,
    Observations=Observations,
)


class TestFitParameters(GenericFOOCTest):
    InputDataModel = FitParametersInput
    OutputDataModel = FittedParameters
    input_data_jsonable = [m["JSONable"] for m in FIT_PARAM_INPUTS_FOOC_TEST]
    output_data_jsonable = [m["JSONable"] for m in FIT_PARAM_OUTPUTS_FOOC_TEST]

    def get_payload_function(self):
        return fit_parameters

    @pytest.fixture(autouse=True)
    def open_meteo_httpserver_fit_parameters(self, open_meteo_httpserver):
        """
        Enable fake Open Meteo endpoint providing data if query parameters
        match the ones expected by fit parameters, i.e. requests data for
        the past 90 days.
        """
        open_meteo_httpserver.add_request(
            # NOTE: lat/lon must match items in `FIT_PARAM_INPUTS_FOOC_TEST`
            lat=49.01365,
            lon=8.40444,
            past_days=90,
        )
