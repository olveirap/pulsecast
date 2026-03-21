# Citations and Attributions

## NYC TLC Trip Record Data

This project uses publicly available trip record data provided by the
New York City Taxi and Limousine Commission (TLC).

- **Source:** NYC TLC Trip Record Data  
  <https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page>
- **Licence:** Public domain / NYC Open Data Terms of Use  
  <https://www.nyc.gov/home/terms-of-use.page>
- **Citation:**  
  New York City Taxi and Limousine Commission (TLC). *TLC Trip Record Data*.
  Published on NYC Open Data. Accessed 2024–2026.

---

## NYC Bus Positions

This project utilizes the NYC Bus Positions dataset, a historical and
real-time archive of bus vehicle positions and trajectories.

- **Source:** NYC Bus Positions Archive (hosted on S3)  
  `s3://nycbuspositions`
- **Licence:** NYC Open Data Terms of Use  
- **Citation:**  
  Metropolitan Transportation Authority (MTA) / NYC Open Data. 
  *NYC Bus Positions Archive*. Published on NYC Open Data and Amazon S3.
  Accessed 2024–2026.

---

## MTA GTFS-Realtime Feed

This project polls the Metropolitan Transportation Authority (MTA)
GTFS-Realtime trip-updates feed for supplementary congestion signals.

- **Source:** MTA Developer Resources – GTFS-Realtime  
  <https://api.mta.info/>
- **Terms of Service:** MTA Developer Terms of Service  
  <https://api.mta.info/terms>
- **Licence:** MTA data is provided free of charge for non-commercial and
  commercial use under the MTA's developer terms.  Attribution is required.
- **Citation:**  
  Metropolitan Transportation Authority (MTA). *GTFS-Realtime Trip Updates*.
  <https://api.mta.info/GTFS>. Accessed 2024–2026.

---

## Software Dependencies

Key open-source libraries used in this project:

| Library | Licence | Reference |
|---|---|---|
| polars | MIT | <https://github.com/pola-rs/polars> |
| lightgbm | MIT | Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree", NeurIPS 2017 |
| pytorch-forecasting | MIT | <https://github.com/jdb78/pytorch-forecasting> |
| statsforecast | Apache-2.0 | <https://github.com/Nixtla/statsforecast> |
| fastapi | MIT | <https://fastapi.tiangolo.com/> |
| mlflow | Apache-2.0 | <https://mlflow.org/> |
| evidently | Apache-2.0 | <https://www.evidentlyai.com/> |
| gtfs-realtime-bindings | Apache-2.0 | <https://github.com/MobilityData/gtfs-realtime-bindings> |
| geopandas | BSD-3-Clause | <https://geopandas.org/> |
| boto3 | Apache-2.0 | <https://github.com/boto/boto3> |
