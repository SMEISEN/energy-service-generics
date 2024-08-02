"""
Datasets nice for testing.

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

import pandas as pd

# Valid input data that can be used to to check the forecasting or optimization
# code (as well as the worker)
REQUEST_INPUTS_FOOC_TEST = [
    {
        "Python": {
            "arguments": {
                "geographic_position": {
                    "latitude": 49.01365,
                    "longitude": 8.40444,
                    "height": 75.3
                }
            },
            "parameters": {
                "autopv_default": {
                    "weights": [
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
                    ],
                    # NOTE: ensemble_fun and peak_power are
                    #       FitParameterArguments  and also FittedParameters!
                    #       That is, they belong to  arguments in
                    #       fit_parameters() and to parameters in
                    #       handle_request()
                    "ensemble_fun": "weighted_sum",
                    "peak_power": 1.,
                    "power_datapoint_id": None
                }
            }
        },
        # NOTE: No difference to Python as no data types included that JSON
        #       cannot handle natively, like datetimes etc.
        "JSONable": {
            "arguments": {
                "geographic_position": {
                    "latitude": 49.01365,
                    "longitude": 8.40444,
                    "height": 75.3
                }
            },
            "parameters": {
                "autopv_default": {
                    "weights": [
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
                    ],
                    # NOTE: ensemble_fun and peak_power are
                    #       FitParameterArguments  and also FittedParameters!
                    #       That is, they belong to  arguments in
                    #       fit_parameters() and to parameters in
                    #       handle_request()
                    "ensemble_fun": "weighted_sum",
                    "peak_power": 1.,
                }
            }
        }
    }
]

# Expected output if if items defined in `REQUEST_INPUT_FOOC_TEST` are
# provided to the forecasting optimization code (or the worker).
"""
The predictions here have been generated with:
    expected_power_pd = AutoPVdefault(
        latitude=49.01365,
        longitude=8.40444,
        altitude=75.3,
        weights": [
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
            ],
        ensemble_fun="weighted_sum",
        peak_power=1.
    ).predict(x=OPEN_METEO_RESPONSE_DF)
    value_message_list_from_series(expected_power_pd)
"""
REQUEST_OUTPUTS_FOOC_TEST = [
    {
        "Python": {
            "power_prediction": [
                {

                    "time": pd.Timestamp("2024-05-03T12:00:00"),
                    'value': 0.3379391719298714
                },
                {

                    "time": pd.Timestamp("2024-05-03T12:15:00"),
                    'value': 0.323054565788261
                },
                {

                    "time": pd.Timestamp("2024-05-03T12:30:00"),
                    'value': 0.3079946537996828
                },
                {

                    "time": pd.Timestamp("2024-05-03T12:45:00"),
                    'value': 0.29240559321003995
                },
                {

                    "time": pd.Timestamp("2024-05-03T13:00:00"),
                    'value': 0.2770945111927875
                },
                {

                    "time": pd.Timestamp("2024-05-03T13:15:00"),
                    'value': 0.2632400098897224
                },
                {
                    "time": pd.Timestamp("2024-05-03T13:30:00"),
                    'value': 0.2514985238504553
                },
                {
                    "time": pd.Timestamp("2024-05-03T13:45:00"),
                    'value': 0.24226226488219454
                },
                {
                    "time": pd.Timestamp("2024-05-03T14:00:00"),
                    'value': 0.23584072766988487
                },
                {
                    "time": pd.Timestamp("2024-05-03T14:15:00"),
                    'value': 0.232403829334564
                },
                {
                    "time": pd.Timestamp("2024-05-03T14:30:00"),
                    'value': 0.22988946550186629
                },
                {
                    "time": pd.Timestamp("2024-05-03T14:45:00"),
                    'value': 0.22755047367372241
                }
            ]
        },
        "JSONable": {
            "power_prediction": [
                {
                    'time': '2024-05-03T12:00:00',
                    'value': 0.3379391719298714
                },
                {
                    'time': '2024-05-03T12:15:00',
                    'value': 0.323054565788261
                },
                {
                    'time': '2024-05-03T12:30:00',
                    'value': 0.3079946537996828
                },
                {
                    'time': '2024-05-03T12:45:00',
                    'value': 0.29240559321003995
                },
                {
                    'time': '2024-05-03T13:00:00',
                    'value': 0.2770945111927875
                },
                {
                    'time': '2024-05-03T13:15:00',
                    'value': 0.2632400098897224
                },
                {
                    'time': '2024-05-03T13:30:00',
                    'value': 0.2514985238504553
                },
                {
                    'time': '2024-05-03T13:45:00',
                    'value': 0.24226226488219454
                },
                {
                    'time': '2024-05-03T14:00:00',
                    'value': 0.23584072766988487
                },
                {
                    'time': '2024-05-03T14:15:00',
                    'value': 0.232403829334564
                },
                {
                    'time': '2024-05-03T14:30:00',
                    'value': 0.22988946550186629
                },
                {
                    'time': '2024-05-03T14:45:00',
                    'value': 0.22755047367372241
                }
            ]
        }
    }
]

# Valid input and output data that should _not_ be used while testing the
# forecasting or optimization code _but only_ for testing the data models.
# This could be useful in case that running the forecasting or optimization
# code is slow or needs mocking, e.g. for data sources. The latter especially,
# as providing realistic data as mock for all cases might cause large efforts.
#
# NOTE: In this example we don't use the input data defined for the FOOC
#       tests to illustrate above that parts of the `GeographicPosition` and
#       'AutoPVdefault' models, i.e. the `height` and `power_datapoint_id` are not
#       necessary to compute the output.
REQUEST_INPUTS_MODEL_TEST = [
    {
        "Python": {
            "arguments": {
                "geographic_position": {
                    "latitude": 49.01365,
                    "longitude": 8.40444,
                    "height": None
                }
            },
            "parameters": {
                "autopv_default": {
                    "weights": [
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
                    ],
                    # NOTE: ensemble_fun and peak_power are
                    #       FitParameterArguments  and also FittedParameters!
                    #       That is, they belong to  arguments in
                    #       fit_parameters() and to parameters in
                    #       handle_request()
                    "ensemble_fun": "weighted_sum",
                    "peak_power": 1.,
                    "power_datapoint_id": None
                }
            }
        },
        # NOTE: No difference to Python as no data types included that JSON
        #       cannot handle natively, like datetimes etc.
        "JSONable": {
            "arguments": {
                "geographic_position": {
                    "latitude": 49.01365,
                    "longitude": 8.40444,
                    "height": None
                }
            },
            "parameters": {
                "autopv_default": {
                    "weights": [
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
                    ],
                    # NOTE: ensemble_fun and peak_power are
                    #       FitParameterArguments  and also FittedParameters!
                    #       That is, they belong to  arguments in
                    #       fit_parameters() and to parameters in
                    #       handle_request()
                    "ensemble_fun": "weighted_sum",
                    "peak_power": 1.,
                    "power_datapoint_id": None
                }
            }
        }
    }
]

REQUEST_OUTPUTS_MODEL_TEST = REQUEST_OUTPUTS_FOOC_TEST + []

# Input data which the request endpoint should reject.
INVALID_REQUEST_INPUTS = [
    # No arguments
    {
        "JSONable": {
            "parameters": {
                "autopv_default": {
                    "weights": [
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
                    ],
                    # NOTE: ensemble_fun and peak_power are
                    #       FitParameterArguments  and also FittedParameters!
                    #       That is, they belong to  arguments in
                    #       fit_parameters() and to parameters in
                    #       handle_request()
                    "ensemble_fun": "weighted_sum",
                    "peak_power": 1.,
                    "power_datapoint_id": None
                }
            }
        }
    },
    # No parameters
    {
        "JSONable": {
            "arguments": {
                "geographic_position": {
                    "latitude": 49.01365,
                    "longitude": 8.40444,
                    "height": None
                }
            }
        }
    },
    # geographic_position not in arguments.
    {
        "JSONable": {
            "arguments": {"geographic_pos": 1.0},
            "parameters": {
                "autopv_default": {
                    "weights": [
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
                    ],
                    # NOTE: ensemble_fun and peak_power are
                    #       FitParameterArguments  and also FittedParameters!
                    #       That is, they belong to  arguments in
                    #       fit_parameters() and to parameters in
                    #       handle_request()
                    "ensemble_fun": "weighted_sum",
                    "peak_power": 1.,
                    "power_datapoint_id": None
                }
            }
        }
    },
    # autopv_default not in parameters
    {
        "JSONable": {
            "arguments": {
                "geographic_position": {
                    "latitude": 49.01365,
                    "longitude": 8.40444,
                    "height": None
                }
            },
            "parameters": {"something_else": "True"}
        }
    }
]

# Output data which output model should reject.
INVALID_REQUEST_OUTPUTS = [
    # No return value.
    {
        "JSONable": None
    },
    # Empty dict returned
    {
        "JSONable": {}
    },
    # values not a value value message list
    {
        "JSONable": {"power_prediction": 0.25}
    }
]

# Test data for the fit parameter endpoints corresponding to the concept
# used for the request endpoint as explained above.
"""
The observations here have been generated with:
    expected_power_pd = AutoPVdefault(
        latitude=49.01365,
        longitude=8.40444,
        altitude=75.3,
        weights": [
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
            ],
        ensemble_fun="weighted_sum",
        peak_power=1.
    ).predict(x=OPEN_METEO_RESPONSE_DF)
    value_message_list_from_series(expected_power_pd)
"""
FIT_PARAM_INPUTS_FOOC_TEST = [
    {
        "Python": {
            "arguments": {
                "geographic_position": {
                    "latitude": 49.01365,
                    "longitude": 8.40444,
                    "height": 75.3
                },
                # NOTE: ensemble_fun and peak_power are FitParameterArguments
                #       and also FittedParameters! That is, they belong to
                #       arguments in fit_parameters() and to parameters in
                #       handle_request()
                "ensemble_fun": "weighted_sum",
                "peak_power": 1.
            },
            "observations": {
                # NOTE: It is necessary to roughly this much of data here as
                #       else the least square fitting might be unstable.
                "measured_power": [
                    {

                        "time": pd.Timestamp("2024-05-03T12:00:00"),
                        'value': 0.3379391719298714
                    },
                    {

                        "time": pd.Timestamp("2024-05-03T12:15:00"),
                        'value': 0.323054565788261
                    },
                    {

                        "time": pd.Timestamp("2024-05-03T12:30:00"),
                        'value': 0.3079946537996828
                    },
                    {

                        "time": pd.Timestamp("2024-05-03T12:45:00"),
                        'value': 0.29240559321003995
                    },
                    {

                        "time": pd.Timestamp("2024-05-03T13:00:00"),
                        'value': 0.2770945111927875
                    },
                    {

                        "time": pd.Timestamp("2024-05-03T13:15:00"),
                        'value': 0.2632400098897224
                    },
                    {
                        "time": pd.Timestamp("2024-05-03T13:30:00"),
                        'value': 0.2514985238504553
                    },
                    {
                        "time": pd.Timestamp("2024-05-03T13:45:00"),
                        'value': 0.24226226488219454
                    },
                    {
                        "time": pd.Timestamp("2024-05-03T14:00:00"),
                        'value': 0.23584072766988487
                    },
                    {
                        "time": pd.Timestamp("2024-05-03T14:15:00"),
                        'value': 0.232403829334564
                    },
                    {
                        "time": pd.Timestamp("2024-05-03T14:30:00"),
                        'value': 0.22988946550186629
                    },
                    {
                        "time": pd.Timestamp("2024-05-03T14:45:00"),
                        'value': 0.22755047367372241
                    }
                ]
            }
        },
        "JSONable": {
            "arguments": {
                "geographic_position": {
                    "latitude": 49.01365,
                    "longitude": 8.40444,
                    "height": 75.3
                },
                # NOTE: ensemble_fun and peak_power are FitParameterArguments
                #       and also FittedParameters! That is, they belong to
                #       arguments in fit_parameters() and to parameters in
                #       handle_request()
                "ensemble_fun": "weighted_sum",
                "peak_power": 1.
            },
            "observations": {
                "measured_power": [
                    {
                        'time': '2024-05-03T12:00:00',
                        'value': 0.3379391719298714
                    },
                    {
                        'time': '2024-05-03T12:15:00',
                        'value': 0.323054565788261
                    },
                    {
                        'time': '2024-05-03T12:30:00',
                        'value': 0.3079946537996828
                    },
                    {
                        'time': '2024-05-03T12:45:00',
                        'value': 0.29240559321003995
                    },
                    {
                        'time': '2024-05-03T13:00:00',
                        'value': 0.2770945111927875
                    },
                    {
                        'time': '2024-05-03T13:15:00',
                        'value': 0.2632400098897224
                    },
                    {
                        'time': '2024-05-03T13:30:00',
                        'value': 0.2514985238504553
                    },
                    {
                        'time': '2024-05-03T13:45:00',
                        'value': 0.24226226488219454
                    },
                    {
                        'time': '2024-05-03T14:00:00',
                        'value': 0.23584072766988487
                    },
                    {
                        'time': '2024-05-03T14:15:00',
                        'value': 0.232403829334564
                    },
                    {
                        'time': '2024-05-03T14:30:00',
                        'value': 0.22988946550186629
                    },
                    {
                        'time': '2024-05-03T14:45:00',
                        'value': 0.22755047367372241
                    }
                ]
            }
        }
    }
]

FIT_PARAM_OUTPUTS_FOOC_TEST = [
    {
        "Python": {
            "autopv_default": {
                "weights": [
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
                ],
                # NOTE: ensemble_fun and peak_power are FitParameterArguments
                #       and also FittedParameters! That is, they belong to
                #       arguments in fit_parameters() and to parameters in
                #       handle_request()
                "ensemble_fun": "weighted_sum",
                "peak_power": 1.,
                "power_datapoint_id": None
            }
        },
        # NOTE: No difference to Python as no data types included that JSON
        #       cannot handle natively, like datetimes etc.
        "JSONable": {
            "autopv_default": {
                "weights": [
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
                ],
                # NOTE: ensemble_fun and peak_power are FitParameterArguments
                #       and also FittedParameters! That is, they belong to
                #       arguments in fit_parameters() and to parameters in
                #       handle_request()
                "ensemble_fun": "weighted_sum",
                "peak_power": 1.,
                "power_datapoint_id": None
            }
        }
    }
]

# Again don't use the FOOC data here, see explanation at comment above
# `REQUEST_INPUTS_MODEL_TEST`.
FIT_PARAM_INPUTS_MODEL_TEST = [
    {
        "Python": {
            "arguments": {
                "geographic_position": {
                    "latitude": 49.01365,
                    "longitude": 8.40444,
                    "height": None
                },
                # NOTE: ensemble_fun and peak_power are FitParameterArguments
                #       and also FittedParameters! That is, they belong to
                #       arguments in fit_parameters() and to parameters in
                #       handle_request()
                "ensemble_fun": "weighted_sum",
                "peak_power": 1.
            },
            "observations": {
                "measured_power": [
                    {
                        "value": None,
                        "time": pd.Timestamp("2024-04-14 13:00:00")
                    },
                    {
                        "value": -0.027931034482758618,
                        "time": pd.Timestamp("2024-04-14 13:15:00")
                    }
                ]
            }
        },
        "JSONable": {
            "arguments": {
                "geographic_position": {
                    "latitude": 49.01365,
                    "longitude": 8.40444,
                    "height": None
                },
                # NOTE: ensemble_fun and peak_power are FitParameterArguments
                #       and also FittedParameters! That is, they belong to
                #       arguments in fit_parameters() and to parameters in
                #       handle_request()
                "ensemble_fun": "weighted_sum",
                "peak_power": 1.
            },
            "observations": {
                "measured_power": [
                    {
                        "value": None,
                        "time": "2024-04-14T13:00:00"
                    },
                    {
                        "value": -0.027931034482758618,
                        "time": "2024-04-14T13:15:00"
                    }
                ]
            }
        }
    }
]

FIT_PARAM_OUTPUTS_MODEL_TEST = FIT_PARAM_OUTPUTS_FOOC_TEST + []

INVALID_FIT_PARAM_INPUTS = [
    # No arguments
    {
        "JSONable": {
            "observations": {
                "measured_power": [
                    {
                        "value": None,
                        "time": "2024-04-14T13:00:00"
                    },
                    {
                        "value": -0.027931034482758618,
                        "time": "2024-04-14T13:15:00"
                    }
                ]
            }
        }
    },
    # No observations
    {
        "JSONable": {
            "arguments": {
                "geographic_position": {
                    "latitude": 49.01365,
                    "longitude": 8.40444,
                    "height": None
                },
                # NOTE: ensemble_fun and peak_power are FitParameterArguments
                #       and also FittedParameters! That is, they belong to
                #       arguments in fit_parameters() and to parameters in
                #       handle_request()
                "ensemble_fun": "weighted_sum",
                "peak_power": 1.
            }
        }
    },
    # no geographic_position in arguments
    {
        "JSONable": {
            "arguments": {"nope": "no coords"},
            "observations": {
                "measured_power": [
                    {
                        "value": None,
                        "time": "2024-04-14T13:00:00"
                    },
                    {
                        "value": -0.027931034482758618,
                        "time": "2024-04-14T13:15:00"
                    }
                ]
            }
        }
    },
    # No measured power in observations
    {
        "JSONable": {
            "arguments": {
                "geographic_position": {
                    "latitude": 49.01365,
                    "longitude": 8.40444,
                    "height": None
                },
                # NOTE: ensemble_fun and peak_power are FitParameterArguments
                #       and also FittedParameters! That is, they belong to
                #       arguments in fit_parameters() and to parameters in
                #       handle_request()
                "ensemble_fun": "weighted_sum",
                "peak_power": 1.
            },
            "observations": [{"some measurements": "not there"}]
        }
    },
    # Measured power not a value message list.
    {
        "JSONable": {
            "arguments": {
                "geographic_position": {
                    "latitude": 49.01365,
                    "longitude": 8.40444,
                    "height": None
                },
                # NOTE: ensemble_fun and peak_power are FitParameterArguments
                #       and also FittedParameters! That is, they belong to
                #       arguments in fit_parameters() and to parameters in
                #       handle_request()
                "ensemble_fun": "weighted_sum",
                "peak_power": 1.
            },
            "observations": {"measured_power": [0.1, 0.2]}
        }
    }
]

INVALID_FIT_PARAM_OUTPUTS = [
    # No return value.
    {
        "JSONable": None
    },
    # Empty dict returned
    {
        "JSONable": {}
    },
    # Fitted AutoPV is empty.
    {
        "JSONable": {"autopv_default": None}
    },
    {
        "JSONable": {"autopv_default": {}}
    }
]

OPEN_METEO_RESPONSE = {
    "latitude": 49.018,
    "longitude": 8.41,
    "generationtime_ms": 51.782965660095215,
    "utc_offset_seconds": 0,
    "timezone": "GMT",
    "timezone_abbreviation": "GMT",
    "elevation": 117.0,
    "minutely_15_units": {
        'time': 'iso8601',
        'shortwave_radiation': 'W/m²',
        'diffuse_radiation': 'W/m²',
        'direct_normal_irradiance': 'W/m²',
        'temperature_2m': '°C',
        'wind_speed_10m': 'km/h',
        'surface_pressure': 'hPa'
    },
    "minutely_15": {
        "time": [
            # This would be more entries in reality but trimmed here.
            '2024-05-03T12:00',
            '2024-05-03T12:15',
            '2024-05-03T12:30',
            '2024-05-03T12:45',
            '2024-05-03T13:00',
            '2024-05-03T13:15',
            '2024-05-03T13:30',
            '2024-05-03T13:45',
            '2024-05-03T14:00',
            '2024-05-03T14:15',
            '2024-05-03T14:30',
            '2024-05-03T14:45'
        ],
        "shortwave_radiation": [
            479.0,
            456.0,
            433.0,
            410.0,
            388.0,
            369.0,
            353.0,
            341.0,
            333.0,
            329.0,
            326.0,
            323.0
        ],
        "diffuse_radiation": [
            332.7,
            327.0,
            319.1,
            309.3,
            298.1,
            286.6,
            275.2,
            264.5,
            254.5,
            245.2,
            235.5,
            225.1
        ],
        "direct_normal_irradiance": [
            175.7,
            156.2,
            139.5,
            125.2,
            113.8,
            106.6,
            103.2,
            104.5,
            110.8,
            122.8,
            138.4,
            157.0
        ],
        "temperature_2m": [
            14.2,
            14.3,
            14.4,
            14.4,
            14.4,
            14.5,
            14.5,
            14.6,
            14.6,
            14.6,
            14.6,
            14.6
        ],
        "wind_speed_10m": [
            20.4,
            20.0,
            19.7,
            19.0,
            18.7,
            18.0,
            17.8,
            17.4,
            16.8,
            16.5,
            15.9,
            15.6
        ],
        "surface_pressure": [
            1001.0,
            1001.1,
            1001.1,
            1001.2,
            1001.2,
            1001.2,
            1001.3,
            1001.3,
            1001.4,
            1001.4,
            1001.4,
            1001.5
        ]
    }
}

OPEN_METEO_RESPONSE_DF = pd.DataFrame(
    index=pd.to_datetime(OPEN_METEO_RESPONSE["minutely_15"]["time"]),
    data={
        "ghi": OPEN_METEO_RESPONSE["minutely_15"]["shortwave_radiation"],
        "dhi": OPEN_METEO_RESPONSE["minutely_15"]["diffuse_radiation"],
        "dni": OPEN_METEO_RESPONSE["minutely_15"]["direct_normal_irradiance"],
        "t2m": OPEN_METEO_RESPONSE["minutely_15"]["temperature_2m"],
        "ws10": OPEN_METEO_RESPONSE["minutely_15"]["wind_speed_10m"],
        "sp": OPEN_METEO_RESPONSE["minutely_15"]["surface_pressure"]
    }
)
