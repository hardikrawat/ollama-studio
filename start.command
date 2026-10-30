#!/bin/bash
# Start local web server for Local Intelligence Workstation
cd "$(dirname "$0")"

PORT=3300
# Find an available port if 3300 is busy
while lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; do
    PORT=$((PORT+1))
done

echo "Starting Local Intelligence Workstation on http://localhost:$PORT ..."
python3 server.py $PORT &
SERVER_PID=$!

sleep 0.8
if command -v open >/dev/null 2>&1; then
    open "http://localhost:$PORT"
elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "http://localhost:$PORT"
fi

# Trap Ctrl+C or terminal exit to terminate background server
trap "kill $SERVER_PID 2>/dev/null; exit" SIGINT SIGTERM EXIT
wait $SERVER_PID
