CDEWS Image Processing and Remote Analytical Inference Service
This repository provides the server-side implementation for the Community-Based Dengue Early Warning System (CDEWS) image processing and inference pipeline. The system functions as a secure, high-availability RESTful service that ingests image submissions from distributed field devices, performs deterministic multi-stage computer vision processing, and transmits structured analytical results to the central CDEWS data infrastructure. It leverages Google Cloud Storage (GCS) for artifact persistence and is optimized for deployment on Google Compute Engine (GCE).

System Overview
The application exposes a POST endpoint (/predict) that accepts authenticated image uploads. Upon receiving an image, the service executes input normalization, secure file handling, binary threshold transformation for morphological analysis, Roboflow-based object detection, generation of annotated visual outputs, local persistence to SQLite and CSV formats, and upload of image artifacts to a GCS bucket. Each uploaded file is assigned a time-limited signed URL to enable secure retrieval by client applications or upstream systems. Processed results are transmitted to the CDEWS central API endpoint for integration with epidemiological monitoring systems.

Google Cloud Storage Integration and Fetching
Authentication is performed via a service account key JSON file on the deployment environment. A storage client is initialized using the service account, and the target bucket is referenced within the service. Artifacts—including original uploads, binary transformations, and annotated outputs—are uploaded using a modular function that accepts a local file path and optional subfolder specification. Clients can fetch uploaded artifacts without direct bucket access using time-limited signed URLs generated from the blob path. These signed URLs provide secure HTTP GET links valid for a configurable expiration interval, allowing programmatic retrieval for dashboards, analytics pipelines, or downstream services without exposing long-term credentials.

Deployment Requirements
Google Compute Engine VM running Ubuntu 22.04, Python 3.9 or later, required Python packages including Flask, OpenCV, Pandas, Requests, Roboflow, google-cloud-storage, and Werkzeug (installable via pip), and a Google Cloud service account key with write access to the configured GCS bucket.

Installation and Deployment
Provision a GCE VM (recommended e2-medium or higher) and ensure inbound TCP traffic on port 5000. Upload all project files to a directory such as /home/server. Place the Google Cloud service account key JSON in the project directory and restrict permissions using chmod 600. Install dependencies using pip3 install -r requirements.txt. Launch the server with python3 server.py; the service will bind to 0.0.0.0:5000 and be externally accessible at http://<vm-external-ip>:5000.

API Specification
Endpoint: POST /predict. Requests must include headers x-api-key and x-api-secret containing the generated authentication credentials. The request body must contain multipart/form-data with a single field named “file” referencing the image to process. The response JSON includes processing timestamp, device code, detected object count, and signed URLs for the original, binary, and annotated images stored in GCS, along with the status of transmission to the central CDEWS API endpoint. Clients can fetch files securely using these signed URLs without service account credentials, and each URL is valid only for a limited duration.

Data Persistence and Artifact Management
Local results are maintained in an SQLite database at /home/server/database/egg_results.db, with a continuously updated CSV export (egg_results.csv) for downstream analysis. All artifacts are uploaded to the configured GCS bucket under structured subfolders, and access is mediated via signed URLs generated at the time of upload, providing secure, time-bounded retrieval for dashboards, integration pipelines, or automated ingestion systems.

Operational Hardening
For production deployments, the service can be configured as a systemd unit to ensure automatic start on machine boot and persistent recovery on failure. The configuration should specify the working directory, Python executable, and a restart policy to provide a resilient runtime environment suitable for continuous ingestion from multiple field devices.

Extensibility
The architecture supports modular extension, allowing integration of additional preprocessing routines, alternative detection models, or customized reporting pipelines without modification to the core processing logic. The service is intended for deployment within the CDEWS ecosystem or research initiatives requiring structured remote image ingestion, inference, and cloud-based artifact management.
