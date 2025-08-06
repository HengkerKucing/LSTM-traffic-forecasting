# Instruksi Deploy LSTM FastAPI ke GKE

## 1. Build & Push Docker Image

```sh
# Build image
 docker build -t asia.gcr.io/final-year-463903/lstm-service:v1 .

# Login ke Google Container Registry (GCR)
gcloud auth configure-docker asia.gcr.io

# Push image ke GCR
docker push asia.gcr.io/final-year-463903/lstm-service:v1
```

## 2. Deploy ke GKE

```sh
# Apply ConfigMap (env DB)
kubectl apply -f k8s/configmap.yaml

# Deploy aplikasi
kubectl apply -f k8s/deployment.yaml

# Buat service internal
kubectl apply -f k8s/service.yaml
```

## 3. Cek Status

```sh
kubectl get pods
kubectl get service lstm-service
```

## 4. Akses Service dari Service Lain (Internal)

Akses endpoint FastAPI dari service lain di cluster:

```
http://lstm-service:8080/predict
http://lstm-service:8080/dbtest
```

## 5. Test Koneksi Database

Akses endpoint `/dbtest` untuk cek koneksi DB:

```
curl http://lstm-service:8080/dbtest
```

## 6. (Opsional) Zip & Upload Project

```sh
zip -r lstm-service.zip . -x "*.git*" "model_lstm_volume.h5" "scaler_volume.save"
# Upload ke Cloud Shell, lalu:
unzip lstm-service.zip -d lstm-service
cd lstm-service
```

---

**Catatan:**
- Service ini hanya bisa diakses internal oleh service lain di cluster (ClusterIP).
- Pastikan DB Cloud SQL bisa diakses dari GKE (public IP atau gunakan Cloud SQL Proxy jika private).