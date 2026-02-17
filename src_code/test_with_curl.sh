curl -X POST "http://127.0.0.1:8000/moderate" \
-H "Content-Type: application/json" \
-d '{
"text": "You are stupid and I hate you",
"user_id": "user42"
}'

