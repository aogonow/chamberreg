# Registration of CAD models to CT scans of ionization chambers.
#
# The scans are not in the image. Download the Zenodo record
# https://doi.org/10.5281/zenodo.22878186 and place its files as
#   data/baseline/config.json, data/baseline/stats.json
#   data/baseline/sinograms/chamber_0_recon_sino.hdf5 ... chamber_9_recon_sino.hdf5
# (the record holds them flat), then mount that directory:
#   docker build -t chamberreg .
#   docker run --rm -v "$PWD/data:/data:ro" -v "$PWD/results:/results" chamberreg \
#       run /data/baseline --out /results --jobs 5 --threads 8
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY chamberreg/ chamberreg/

ENTRYPOINT ["python", "-m", "chamberreg"]
CMD ["--help"]
