# curl -X POST "http://127.0.0.1:8000/moderate_image" -F "file=@./hate.png"
#curl -X POST "http://127.0.0.1:8000/moderate_image" -F "file=@./riot.png"
curl -X POST "http://127.0.0.1:8000/moderate_image" -F "file=@./bits.jpg"
