"""
Forecasting or optimization code of the service.
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
from typing import Union

import pvlib
import requests
import numpy as np
import pandas as pd

from esg.utils.pandas import series_from_value_message_list
from esg.utils.pandas import value_message_list_from_series
from scipy.optimize import least_squares

OPEN_METEO_API_URL_DEFAULT = "https://api.open-meteo.com"


def fetch_meteo_data(lat, lon, past_days=0):
    """
    Fetch relevant NWP data for the PV forecast from Open Meteo API.

    Arguments:
    ----------
    lat : float
        Latitude in degree.
    lon : float
        Longitude in degree.
    past_days : int
        Number of days in past for which the Forecast should be computed too.
        Up to 90 days is supported by the API. Use for fitting parameters.

    Environment Variables:
    ----------------------
    OPEN_METEO_API_URL:
        The Base URL of the Open Meteo API. Option is used in tests.
        Defaults to to `OPEN_METEO_API_URL_DEFAULT`.

    Returns:
    --------
    meteo_data : pd.DataFrame
        Irradiance data in format expected by pvlib.
    """

    open_meteo_api_url = os.getenv("OPEN_METEO_API_URL")
    if not open_meteo_api_url:
        open_meteo_api_url = OPEN_METEO_API_URL_DEFAULT

    # Fetch the data from Open Meteo API in their JSON format.
    response = requests.get(
        f"{open_meteo_api_url}/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "minutely_15": (
                "shortwave_radiation,"
                "diffuse_radiation,"
                "direct_normal_irradiance,"
                "temperature_2m,"
                "wind_speed_10m,"
                "surface_pressure"
            ),
            "past_days": past_days,
            "forecast_days": 1,
        },
    )
    response.raise_for_status()
    meteo_data_json = response.json()["minutely_15"]

    # Convert to pandas DataFrame
    meteo_data = pd.DataFrame(
        index=pd.to_datetime(meteo_data_json.pop("time")), data=meteo_data_json
    )
    name_map = {
        "shortwave_radiation": "ghi",
        "diffuse_radiation": "dhi",
        "direct_normal_irradiance": "dni",
        "temperature_2m": "t2m",
        "wind_speed_10m": "ws10",
        "surface_pressure": "sp"
    }
    meteo_data.rename(name_map, axis=1, inplace=True)

    return meteo_data


class AutoPVdefault:

    def __init__(self,
                 latitude: Union[int, float], longitude: Union[int, float],
                 altitude: Union[int, float] = None, weights: list = None,
                 peak_power: Union[int, float] = 1.,
                 ensemble_fun: str = "weighted_sum",
                 tilt_values: list = None, azimuth_values: list = None,
                 pv_module=None, inverter=None, temperature_model_params=None,
                 irradiance_model: str = "DISC"):
        """
        AutoPV with the default model pool based on physical-inspired modeling.
        The underlying idea of AutoPV is to describe the arbitrary mounting
        configuration of a new PV plant as a convex linear combination of
        outputs from a sufficiently diverse ensemble pool of PV models of the
        same region. AutoPV incorporates three steps: i) create the ensemble
        model pool, ii) form the ensemble output by an optimally weighted sum
        of the scaled model outputs in the pool, and iii) rescale the ensemble
        output with the new PV plant’s peak power rating.

        Paper:
        https://doi.org/10.1145/3575813.3597348

        Standalone implementation:
        https://github.com/SMEISEN/AutoPV

        Arguments:
        ----------
        latitude : Union[int, float]
            Latitude coordinate of the PV plant.
        longitude : Union[int, float]
            Longitude coordinate of the PV plant.
        altitude : Union[int, float], optional
            Altitude of the PV plant.
        weights : list, optional
            Weights representing the contribution of the ensemble pool models.
        peak_power : List, optional
            Peak power rating of the PV plant.
        ensemble_fun : str, optional
            Objective function to fit the ensemble weights. Options:
            'weighted_sum':
                Weighted sum, estimates peak power, constraints:
                w >= 0
            'weighted_average':
                Weighted average, does not estimate peak power, constraints:
                w in [0,1], sum(w) = 1
            Default is 'weighted_sum'.
        tilt_values : list, optional
            List of tilt values to be considered in the ensemble model pool.
        azimuth_values : list, optional
            List of azimuth values to be considered in the ensemble model pool.
        pv_module : optional
            PV module model (pvlib).
        inverter : optional
            Inverter model (pvlib).
        temperature_model_params : optional
            Temperature model params (pvlib).
        irradiance_model : str, optional
            Irradiance model. Options: 'DISC', 'Erbs'. Default is 'DISC'.
        """

        super().__init__()
        if tilt_values is None:
            tilt_values = [15, 45, 75]
        if azimuth_values is None:
            azimuth_values = [0, 90, 180, 270]
        if pv_module is None:
            pv_module = pvlib.pvsystem.retrieve_sam('SandiaMod')[
                'Canadian_Solar_CS5P_220M___2009_']
        if inverter is None:
            inverter = pvlib.pvsystem.retrieve_sam('cecinverter')[
                'ABB__MICRO_0_25_I_OUTD_US_208__208V_']
        if temperature_model_params is None:
            temperature_model_params = \
                pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm'][
                    'open_rack_glass_glass']

        self._implemented_ensemble_functions = ["weighted_sum",
                                                "weighted_average"]
        if ensemble_fun not in self._implemented_ensemble_functions:
            raise NotImplementedError(
                f"Ensemble function {ensemble_fun} is not implemented!"
                f"Implemented ensemble functions: "
                f"{self._implemented_ensemble_functions}")
        self._implemented_irradiance_models = ["Erbs", "DISC"]
        if irradiance_model not in self._implemented_irradiance_models:
            raise NotImplementedError(
                f"Irradiance model {self._implemented_irradiance_models} "
                f"is not implemented!"
                f"Implemented models: {self._implemented_irradiance_models}")

        self.tilt_values = tilt_values
        self.azimuth_values = azimuth_values
        self.lat = latitude
        self.lon = longitude
        self.weights = weights
        self.peak_power = peak_power
        self.ensemble_fun = ensemble_fun
        self.altitude = altitude
        self.pv_module = pv_module
        self.inverter = inverter
        self.temperature_model_params = temperature_model_params
        self.irradiance_model = irradiance_model

        self.weights_ = self.weights
        self.peak_power_ = self.peak_power

    def fit(self, X: Union[pd.DataFrame, pd.Series],
            y: Union[pd.DataFrame, pd.Series]) -> None:
        """
        Predict with the AutoPV model.

        Arguments:
        ----------
        X : Union[pd.DataFrame, pd.Series]
             The inputs for the physical-inspired PV model.
        y : Union[pd.DataFrame, pd.Series]
            The realized values, used to fit the ensemble.

        Returns:
        --------
        None
        """

        model_pool_predictions = self._get_model_pool_predictions(X=X)

        self._fit_ensemble(X=model_pool_predictions, y=y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict with the AutoPV model.

        Arguments:
        ----------
        X : Union[pd.DataFrame, pd.Series]
             The inputs for the physical-inspired PV model.

        Returns:
        --------
        predictions : pd.Series
             The AC power generation.
        """

        model_pool_predictions = self._get_model_pool_predictions(X=X)

        if self.ensemble_fun == "weighted_sum":
            result = np.dot(model_pool_predictions.values, self.weights_)
        elif self.ensemble_fun == "weighted_average":
            result = np.average(model_pool_predictions.values, axis=1,
                                weights=self.weights_) * self.peak_power

        return pd.Series(data=result, index=X.index)

    def _pv_array(self, surface_tilt: Union[int, float],
                  surface_azimuth: Union[int, float], **kwargs) -> pd.Series:
        """
        Create the physical-inspired modeling pipeline of the PV array.

        Arguments:
        ----------
        surface_tilt : Union[int, float]
             Tilt angle of the PV array.
        surface_azimuth : Union[int, float]
            Azimuth angle of the PV array.
        kwargs
            Inputs of the model,
            required: "ghi", "temperature", "wind_speed",
            optional: "pressure", "dni", "dhi"

        Returns:
        --------
        ac : pd.Series
             The AC power generation in W.
        """

        arguments = ["ghi", "t2m", "ws10"]
        for argument in arguments:
            if argument not in kwargs.keys():
                raise SyntaxError(
                    f"The PV array model miss {argument} as input. The module "
                    f"needs {arguments} as input. {kwargs} are given as input."
                    f"Add {argument}=<desired_input> when adding PV array "
                    f"model to the pipeline."
                )
        ghi = kwargs.get("ghi")
        dni = kwargs.get("dni")
        dhi = kwargs.get("dhi")
        temperature = kwargs.get("t2m")
        wind_speed = kwargs.get("ws10")
        pressure = kwargs.get("sp")

        ghi.index = ghi.index.tz_localize(None)
        temperature.index = temperature.index.tz_localize(None)
        wind_speed.index = wind_speed.index.tz_localize(None)
        if pressure is not None:
            pressure.index = pressure.index.tz_localize(None)
        if dni is not None:
            dni.index = dni.index.tz_localize(None)
        if dhi is not None:
            dhi.index = dhi.index.tz_localize(None)

        system = {
            'module': self.pv_module,
            'inverter': self.inverter,
            'surface_azimuth': surface_azimuth,
            "surface_tilt": surface_tilt
        }

        solpos = pvlib.solarposition.get_solarposition(
            time=ghi.index,  # required
            latitude=self.lat,  # required
            longitude=self.lon,  # required
            altitude=self.altitude,  # default None
            temperature=temperature,  # default 12
            pressure=pressure,  # default None
        )

        if dni is None and dhi is None:
            if self.irradiance_model == "DISC":
                dni = pvlib.irradiance.disc(
                    ghi=ghi,  # required
                    solar_zenith=solpos['apparent_zenith'],  # required
                    datetime_or_doy=ghi.index,  # required
                    pressure=pressure if pressure is not None else 101325
                    # default 101325
                )["dni"]
                dhi = ghi - dni * pvlib.tools.cosd(solpos['apparent_zenith'])
            elif self.irradiance_model == "Erbs":
                res = pvlib.irradiance.erbs(
                    ghi=ghi,  # required
                    zenith=solpos['apparent_zenith'],  # required
                    datetime_or_doy=ghi.index,  # required
                )
                dni = res["dni"]
                dhi = res["dhi"]
            dhi[dhi < 0] = 0  # cutoff negative values
        elif dni is None and dhi is not None:
            dhi = dhi
            dni = (ghi - dhi) / pvlib.tools.cosd(solpos['apparent_zenith'])
            dni[dni < 0] = 0  # cutoff negative values
        elif dni is not None and dhi is None:
            dni = dni
            dhi = ghi - dni * pvlib.tools.cosd(solpos['apparent_zenith'])
            dni[dni < 0] = 0  # cutoff negative values

        dni_extra = pvlib.irradiance.get_extra_radiation(
            datetime_or_doy=ghi.index
        )
        airmass_rel = pvlib.atmosphere.get_relative_airmass(
            zenith=solpos['apparent_zenith']
        )
        airmass_abs = pvlib.atmosphere.get_absolute_airmass(
            airmass_relative=airmass_rel,  # required
            pressure=pressure if pressure is not None else 101325
            # default 101325
        )
        aoi = pvlib.irradiance.aoi(
            surface_tilt=system['surface_tilt'],  # required
            surface_azimuth=system['surface_azimuth'],  # required
            solar_zenith=solpos["apparent_zenith"],  # required
            solar_azimuth=solpos["azimuth"],  # required
        )

        total_irradiance = pvlib.irradiance.get_total_irradiance(
            surface_tilt=system['surface_tilt'],  # required
            surface_azimuth=system['surface_azimuth'],  # required
            solar_zenith=solpos['apparent_zenith'],  # required
            solar_azimuth=solpos['azimuth'],  # required
            dni=dni,  # required
            ghi=ghi,  # required
            dhi=dhi,  # required
            dni_extra=dni_extra,  # default None
            airmass=airmass_rel,  # default None
            model='haydavies',
        )
        cell_temperature = pvlib.temperature.sapm_cell(
            poa_global=total_irradiance['poa_global'],  # required
            temp_air=temperature,  # required
            wind_speed=wind_speed,  # required
            **self.temperature_model_params,  # required
        )
        effective_irradiance = pvlib.pvsystem.sapm_effective_irradiance(
            poa_direct=total_irradiance['poa_direct'],  # required
            poa_diffuse=total_irradiance['poa_diffuse'],  # required
            airmass_absolute=airmass_abs,  # required
            aoi=aoi,  # required
            module=self.pv_module,  # required
        )
        dc = pvlib.pvsystem.sapm(
            effective_irradiance=effective_irradiance,  # required
            temp_cell=cell_temperature,  # required
            module=self.pv_module  # required
        )
        ac = pvlib.inverter.sandia(
            v_dc=dc['v_mp'],  # required
            p_dc=dc['p_mp'],  # required
            inverter=self.inverter  # required
        )

        return ac

    def _get_model_pool_predictions(self, X: Union[
        pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """
        Create the ensemble model pool based on the specified tilt-azimuth grid.

        Arguments:
        ----------
        X : Union[pd.DataFrame, pd.Series]
             The inputs for the physical-inspired PV model.
        y : Union[pd.DataFrame, pd.Series]
            The realized values, used to fit the ensemble.

        Returns:
        --------
        predictions : pd.DataFrame
             The ensemble model pool predictions.
        """

        predictions = {}
        for tilt in self.tilt_values:
            for azimuth in self.azimuth_values:
                predictions[f"tilt{tilt}_azimuth{azimuth}"] = self._pv_array(
                    surface_tilt=tilt, surface_azimuth=azimuth, **X).clip(
                    lower=0) / 220  # peak power rating of array

        return pd.DataFrame(predictions)

    def _fit_ensemble(self, X: Union[pd.DataFrame, pd.Series],
                      y: Union[pd.DataFrame, pd.Series]) -> None:
        """
        Fit the ensemble.

        Arguments:
        ----------
        X : Union[pd.DataFrame, pd.Series]
             The forecasts that are ensembled.
        y : Union[pd.DataFrame, pd.Series]
            The realized values, used to fit the ensemble.

        Returns:
        --------
        None
        """

        N = len(X.columns)
        bounds = [[0.0 for _ in range(N)]]
        x0 = [1 / N] * N
        kwargs = {"p_n": X.values, "t": y.values}

        if self.ensemble_fun == "weighted_sum":
            bounds += [[np.inf for _ in range(N)]]
            fun = self._assess_weighted_sum
        elif self.ensemble_fun == "weighted_average":
            bounds += [[1.0 for _ in range(N)]]
            fun = self._assess_weighted_average
            y /= self.peak_power

        result = least_squares(fun=fun, x0=x0, bounds=bounds, kwargs=kwargs,
                               ftol=1e-12, xtol=1e-12, gtol=1e-12)

        if self.ensemble_fun == "weighted_sum":
            self.weights_ = result.x
            self.peak_power_ = sum(result.x)
        elif self.ensemble_fun == "weighted_average":
            self.weights_ = self._normalize_weights(result.x)

    @staticmethod
    def _assess_weighted_sum(x: np.array, p_n: np.array, t: np.array) -> float:
        """
        Assesses the weights of a trial using the weighted sum and returns the
        Mean Squared Error (MSE).

        Arguments:
        ----------
        x : np.array
            The variable to be optimized.
        p_n : np.array
            The forecasts that are ensembled.
        t : np.array
            The realized values, used to fit the ensemble.

        Returns:
        --------
        mse : float
            The MSE of the ensemble.
        """

        p = np.dot(p_n, x)

        return float(np.mean((p - t) ** 2))

    @staticmethod
    def _assess_weighted_average(x: np.array, p_n: np.array,
                                 t: np.array) -> float:
        """
        Assesses the weights of a trial using the weighted average and returns
        the Mean Squared Error (MSE).

        Arguments:
        ----------
        x : np.array
            The variable to be optimized.
        p_n : np.array
            The forecasts that are ensembled.
        t : np.array
            The realized values, used to fit the ensemble.

        Returns:
        --------
        mse : float
            The MSE of the ensemble.
        """

        p = np.average(p_n, axis=1, weights=x)

        return float(np.mean((p - t) ** 2))

    @staticmethod
    def _normalize_weights(weights: np.array) -> np.array:
        """
        Normalizes the weights in the range [0,1]

        Arguments:
        ----------
        weights : np.array
            The weights to be normalized.

        Returns:
        --------
        weights : np.array
            The normalized weights.
        """

        return np.array([weight / sum(weights) for weight in weights])


def handle_request(input_data) -> dict:
    """
    Compute the AutoPV prediction.

    Arguments:
    ----------
    input_data :
        Inputs required for the AutoPV prediction.

    Returns:
    --------
    output_data : dict
        The AutoPV prediction.
    """
    arguments = input_data.arguments
    parameters = input_data.parameters
    lat = getattr(arguments.geographic_position, "latitude")
    lon = getattr(arguments.geographic_position, "longitude")
    altitude = getattr(arguments.geographic_position, "height", None)
    weights = getattr(parameters.autopv_default, "weights")
    ensemble_fun = getattr(parameters.autopv_default, "ensemble_fun", "weighted_sum")
    peak_power = getattr(parameters.autopv_default, "peak_power", 1.)

    meteo_data = fetch_meteo_data(
        lat=lat,
        lon=lon,
    )

    pv_power = AutoPVdefault(
        latitude=lat,
        longitude=lon,
        altitude=altitude,
        weights=weights,
        ensemble_fun=ensemble_fun,
        peak_power=peak_power
    ).predict(X=meteo_data)

    output_data = {"power_prediction": value_message_list_from_series(pv_power)}
    return output_data


def fit_parameters(input_data) -> dict:
    """
    Fit the AutoPV model.

    Arguments:
    ----------
    input_data :
        Inputs required for fitting the AutoPV model.

    Returns:
    --------
    parameters : dict
        The AutoPV parameters.
    """

    arguments = input_data.arguments
    lat = getattr(arguments.geographic_position, "latitude")
    lon = getattr(arguments.geographic_position, "longitude")
    altitude = getattr(arguments.geographic_position, "height", None)
    ensemble_fun = getattr(arguments, "ensemble_fun", "weighted_sum")
    peak_power = getattr(arguments, "peak_power", 1.)

    meteo_data = fetch_meteo_data(
        lat=lat,
        lon=lon,
        past_days=90,
    )
    measured_power = series_from_value_message_list(
        input_data.observations.measured_power
    )
    idx = meteo_data.index.intersection(measured_power.index)

    autopv = AutoPVdefault(
        latitude=lat,
        longitude=lon,
        altitude=altitude,
        ensemble_fun=ensemble_fun,
        peak_power=peak_power
    )
    autopv.fit(X=meteo_data.loc[idx], y=measured_power.loc[idx])

    parameters = {
        "autopv_default": {
            "weights": autopv.weights_.tolist(),
            "ensemble_fun": autopv.ensemble_fun,
            "peak_power": float(autopv.peak_power_)
        }
    }

    return parameters
