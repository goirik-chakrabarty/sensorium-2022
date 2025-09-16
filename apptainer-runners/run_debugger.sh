# cd /notebooks
# python train_sensorium.py

export SSL_CERT_FILE=/project/wandb_certs.pem
cd /notebooks
echo "Waiting for debugger to attach on port 5678..."
# python -m debugpy --listen 0.0.0.0:5678 --wait-for-client train_sensorium.py
python train_sensorium.py
# python /mnt/vast-react/projects/agsinz_foundation_model_brain/goirik/curriculum-learning/sensorium-2022/notebooks/generate_gamma_fits.py