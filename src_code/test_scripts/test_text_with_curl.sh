curl -X POST "http://127.0.0.1:8000/moderate" \
-H "Content-Type: application/json" \
-d '{
"text": "You are intelligent and smart",
"user_id": "user42"
}'

