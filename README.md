CDEWS Image Processing and Remote Analytical Inference Service

This repository provides the server-side implementation for the Community-Based Dengue Early Warning System (CDEWS) remote image analysis pipeline. The system operates as a high-availability inference endpoint designed to process image submissions from distributed field devices, execute multi-stage computer vision operations, and return structured analytical outputs to the upstream CDEWS data collection infrastructure. The service is optimized for deployment on Google Compute Engine (GCE) and leverages Google Cloud Storage (GCS) for resilient object persistence.

System Overview

The application exposes a RESTful endpoint designed to receive authenticated image uploads from remote devices. Upon receiving a submission, the service performs a deterministic processing workflow consisting of input normalization, binary threshold transformation, Roboflow-based object detection, annotated output generation, structured database persistence, GCS artifact upload, and telemetry reporting to the central CDEWS platform. Each processing stage is executed to maintain reproducibility, auditability, and consistent analytical precision across heterogeneous device deployments.

Deployment Requirements

The system requires Python 3.9+ on an Ubuntu 22.04 Google Compute Engine instance with inbound TCP traffic permitted on port 5000. The environment must include Flask, OpenCV, Pandas, Requests, Roboflow, and google-cloud-storage. All dependencies are installable via the requirements.txt file included in this repository. A Google Cloud service account key with write access to the designated storage bucket must be placed at /home/server/service_key.json.

Deployment Procedure

Provision a GCE virtual machine (recommended e2-medium or higher), enable HTTP/HTTPS access, and open firewall port 5000.2. Upload all project files to /home/server on the VM and ensure that the service account key file is present and permission-restricted using chmod 600.3. Install all required Python libraries by executing pip3 install -r requirements.txt.4. Launch the server with python3 server.py. The API service will bind to 0.0.0.0:5000 and become accessible via http://<vm-external-ip>:5000.

API Specification

Endpoint: POST /predict. Requests must include the headers x-api-key and x-api-secret containing the generated credentials. The request body must be multipart/form-data containing a single field named “file” referencing the image to be analyzed. The service returns a JSON response with the processing timestamp, device code, detection count, and signed URLs referencing the original, binary-transformed, and annotated images stored in GCS, along with the upstream response from the CDEWS data collection endpoint.

Data Persistence Model

All operational records are stored locally within an SQLite database at /home/server/database/egg_results.db. A continuously updated CSV representation (egg_results.csv) is produced in the same directory to support simplified downstream integration with analytical or visualization systems. Image artifacts generated during the pipeline are uploaded to the configured Google Cloud Storage bucket. Access to these objects is controlled by expiring signed URLs generated at inference time to maintain secure, time-bounded retrieval.

Operational Integration

The service can be elevated to a long-running system process through the creation of a systemd unit file, enabling full automatic startup on machine boot and persistent recovery if the process terminates unexpectedly. This configuration is recommended for production deployments and environments requiring uninterrupted field device connectivity.

Additional Information

This repository is intended for technical deployments within the CDEWS ecosystem or within research projects requiring structured remote image ingestion, computer vision inference, and cloud-backed artifact management. The architecture supports extensibility, allowing integration of additional preprocessing steps, alternative detection models, or custom reporting pipelines with minimal modifications to the underlying service logic.
