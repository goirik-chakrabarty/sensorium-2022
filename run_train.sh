# cd /notebooks
# python train_sensorium.py

export SSL_CERT_FILE=/project/wandb_certs.pem
cd /notebooks
echo "Waiting for debugger to attach on port 5678..."
python -m debugpy --listen 0.0.0.0:5678 --wait-for-client train_sensorium.py