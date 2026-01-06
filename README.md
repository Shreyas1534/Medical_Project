This repository contains the TensorFlow-based medical image inference service for Auralysis, deployed on Render.
It serves as a standalone ML prediction API that returns disease predictions and confidence scores to the Railway backend.

What This Service Does

This service:

---Loads a trained TensorFlow SavedModel

---Accepts MRI or X-ray images via API

---Performs inference using the model

Returns:

---predicted disease class

---confidence score

---full class probability distribution

This API is consumed by the Railway backend, which handles reporting, summarization, voice output, and monitoring.
