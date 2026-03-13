#!/bin/bash
set -eu

wait_for_server() {
  local port=$1
  local max_retries=60
  local wait_seconds=5

  echo "Waiting for server to start on port $port..."

  for ((i=1; i<=max_retries; i++)); do
    if curl --fail --silent --output /dev/null "http://localhost:${port}/ping"; then
      echo "Server is up and running!"
      return 0
    fi

    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "Server process (PID $SERVER_PID) has died unexpectedly."
        return 1
    fi

    echo "Attempt $i/$max_retries: Server not ready yet. Retrying in ${wait_seconds}s..."
    sleep $wait_seconds
  done

  echo "Timeout: Server did not start within $((max_retries * wait_seconds)) seconds."
  return 1
}


echo "Create Input Files."
# [TAG:CreateFiles]
cat << EOS > params.csv
1.00,0.00
0.75,0.25
0.50,0.50
0.25,0.75
0.00,1.00
EOS
cat << EOS > values.csv
0.00,1.00
3.00,2.00
4.00,5.00
7.00,6.00
8.00,9.00
EOS
# [TAG:CreateFiles_End]


echo "Train a Bezier Simplex Model with a package running."
# [TAG:RunPackageTraining]
python -m torch_bsf \
  --params params.csv \
  --values values.csv \
  --meshgrid params.csv \
  --degree 3
# [TAG:RunPackageTraining_End]

echo "Train a Bezier Simplex Model with MLflow."
# [TAG:RunMLflowTraining]
mlflow run https://github.com/NaokiHamada/pytorch-bsf \
  -P params=params.csv \
  -P values=values.csv \
  -P meshgrid=params.csv \
  -P degree=3
# [TAG:RunMLflowTraining_End]

# [TAG:FetchLatestRunID]
LATEST_RUN_ID=$(ls -td mlruns/0/*/ | grep -v "models" | head -1 | xargs basename)
echo "Tested Run ID: ${LATEST_RUN_ID}"
# [TAG:FetchLatestRunID_End]

echo "Local Prediction."
# [TAG:MakePrediction]
mlflow models predict \
  --env-manager=conda \
  --model-uri "runs:/${LATEST_RUN_ID}/model" \
  --content-type csv \
  --input-path params.csv \
  --output-path test_values.json
cat test_values.json
# result will be shown like {"predictions": [{"0": 0.05797366052865982, ...}
# [TAG:MakePrediction_End]


echo "Web API Prediciton."
# [TAG:ServeAPI]
mlflow models serve \
  --env-manager=conda \
  --model-uri "runs:/${LATEST_RUN_ID}/model" \
  --host localhost \
  --port 5001 &
# [TAG:ServeAPI_End]

SERVER_PID=$!
trap 'kill "$SERVER_PID"' EXIT

echo "Waiting for server to start..."
if ! wait_for_server 5001; then
  exit 1
fi

# [TAG:PredictWithHTTPPost]
curl http://localhost:5001/invocations -H 'Content-Type: application/json' -d '{
  "dataframe_split":{
    "columns": ["t1", "t2"],
    "data": [
       [0.2, 0.8],
       [0.7, 0.3]
    ]
  }
}'
# [TAG:PredictWithHTTPPost_End]
