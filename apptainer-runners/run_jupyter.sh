# cd /notebooks
# python train_sensorium.py

export SSL_CERT_FILE=/project/wandb_certs.pem
cd /notebooks
jupyter lab --allow-root --ip=0.0.0.0 --no-browser --port=8888 --NotebookApp.token='1234' --notebook-dir='/project'