# Smart Crop Productivity Optimizer

This project presents a machine learning system designed to analyze agricultural conditions and estimate crop productivity under different soil and environmental conditions.

The system uses agricultural and environmental features to provide data-driven insights that can support crop productivity planning and optimization.

## Dataset

The project uses agricultural data containing:

* Nitrogen (N)
* Phosphorus (P)
* Potassium (K)
* Temperature
* Humidity
* Rainfall
* Soil pH

The target variable represents estimated crop productivity.

## Model

A **Random Forest Regressor** is used to estimate crop productivity.

The model learns relationships between soil nutrients, environmental conditions, and expected productivity.

## Performance

The model achieved approximately:

* **R² Score:** 0.92
* **Mean Squared Error:** 55.91

These results were obtained using the reconstructed prototype dataset.

## Development Tools

Python
Pandas
NumPy
Scikit-learn
Random Forest Regressor
Machine Learning

## Future Development

Future improvements may include integrating real-time weather data, soil sensor measurements, irrigation information, and additional crop-specific features to improve productivity optimization.
