# Participant testing Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Copy participant code
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the agent runs on
EXPOSE 5008

# Set environment variable for the port
ENV PORT=5008

CMD ["python", "agent.py"]
