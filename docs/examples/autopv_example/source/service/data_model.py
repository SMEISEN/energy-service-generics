"""
Definition of the data models for de-/serializing data and the docs.

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
from typing import Optional, Union

from esg.models.base import _BaseModel
from esg.models.datapoint import ValueMessageList
from esg.models.metadata import GeographicPosition
from pydantic import Field


class AutoPV(_BaseModel):
    weights: list = Field(
        examples=[[
            0.08333333333333333,
            0.08333333333333333,
            0.08333333333333333,
            0.08333333333333333,
            0.08333333333333333,
            0.08333333333333333,
            0.08333333333333333,
            0.08333333333333333,
            0.08333333333333333,
            0.08333333333333333,
            0.08333333333333333,
            0.08333333333333333
        ]],
        description=(
            "The weights define the contribution of the models "
            "in the default ensemble pool, e.g. with the weights "
            "[0,0,0,0,0,0,0,1,0,0,0,0] only the 8th model in the "
            "pool contributes to the forecast, which is a model "
            "representing South 45°."
        )
    )
    ensemble_fun: Optional[str] = Field(
        examples=['weighted_sum'],
        description=(
            "Objective function to fit the ensemble weights.Options:"
            "'weighted_sum':"
            "Weighted sum, estimates peak power, constraints:"
            "w >= 0"
            "'weighted_average':"
            "Weighted average, does not estimate peak power, constraints:"
            "w in[0, 1], sum(w) = 1"
            "Default is 'weighted_sum'."
        )
    )
    peak_power: Optional[Union[int, float]] = Field(
        examples=[1.],
        ge=0.,
        description=(
            "The peak power is a quantity specified in the "
            "data sheet of the PV module and measured at "
            "Standard Test Conditions (STC) by the manufacturer. "
            "The unit of the peak power is kWp."
            "Required input for ensemble_fun='weighted_average'"
            "Ignored input for ensemble_fun='weighted_sum'"
        )
    )
    power_datapoint_id: Optional[int] = Field(
        None,
        examples=[1],
        description=(
            "The id of the datapoint which is used to store forecasts "
            "of power production and measurements of the same, at least "
            "if such measurements exist."
        )
    )


class RequestArguments(_BaseModel):
    geographic_position: GeographicPosition


class RequestOutput(_BaseModel):
    power_prediction: ValueMessageList = Field(
        description="Prediction of power production in W"
    )


class FittedParameters(_BaseModel):
    autopv_default: AutoPV


class Observations(_BaseModel):
    measured_power: ValueMessageList = Field(
        description="Measured power production in W",
        default=[
            {
                "value": 0.00000,
                "time": "2024-07-29T00:00:00",
            },
            {
                "value": 0.00000,
                "time": "2024-07-29T01:00:00",
            },
            {
                "value": 0.00000,
                "time": "2024-07-29T02:00:00",
            },
            {
                "value": 0.00000,
                "time": "2024-07-29T03:00:00",
            },
            {
                "value": 0.00000,
                "time": "2024-07-29T04:00:00",
            },
            {
                "value": 0.02376,
                "time": "2024-07-29T05:00:00",
            },
            {
                "value": 0.08683,
                "time": "2024-07-29T06:00:00",
            },
            {
                "value": 0.26627,
                "time": "2024-07-29T07:00:00",
            },
            {
                "value": 0.45924,
                "time": "2024-07-29T08:00:00",
            },
            {
                "value": 0.62310,
                "time": "2024-07-29T09:00:00",
            },
            {
                "value": 0.73314,
                "time": "2024-07-29T10:00:00",
            },
            {
                "value": 0.77704,
                "time": "2024-07-29T11:00:00",
            },
            {
                "value": 0.78298,
                "time": "2024-07-29T12:00:00",
            },
            {
                "value": 0.72768,
                "time": "2024-07-29T13:00:00",
            },
            {
                "value": 0.62327,
                "time": "2024-07-29T14:00:00",
            },
            {
                "value": 0.46899,
                "time": "2024-07-29T15:00:00",
            },
            {
                "value": 0.28192,
                "time": "2024-07-29T16:00:00",
            },
            {
                "value": 0.10024,
                "time": "2024-07-29T17:00:00",
            },
            {
                "value": 0.03100,
                "time": "2024-07-29T18:00:00",
            },
            {
                "value": 0.00000,
                "time": "2024-07-29T19:00:00",
            },
            {
                "value": 0.00000,
                "time": "2024-07-29T20:00:00",
            },
            {
                "value": 0.00000,
                "time": "2024-07-29T21:00:00",
            },
            {
                "value": 0.00000,
                "time": "2024-07-29T22:00:00",
            },
            {
                "value": 0.00000,
                "time": "2024-07-29T23:00:00",
            }
        ]
    )


class FitParameterArguments(_BaseModel):
    geographic_position: GeographicPosition
    ensemble_fun: Optional[str] = Field(
        examples=['weighted_sum'],
        description=(
            "Objective function to fit the ensemble weights.Options:"
            "'weighted_sum':"
            "Weighted sum, estimates peak power, constraints:"
            "w >= 0"
            "'weighted_average':"
            "Weighted average, does not estimate peak power, constraints:"
            "w in[0, 1], sum(w) = 1"
            "Default is 'weighted_sum'."
        )
    )
    peak_power: Optional[Union[int, float]] = Field(
        examples=[1.],
        ge=0.,
        description=(
            "The peak power is a quantity specified in the "
            "data sheet of the PV module and measured at "
            "Standard Test Conditions (STC) by the manufacturer. "
            "The unit of the peak power is kWp."
            "Required input for ensemble_fun='weighted_average'"
            "Ignored input for ensemble_fun='weighted_sum'"
        )
    )
