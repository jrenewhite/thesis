import socket
import threading
import logging
import os

COORDINATOR_PORT = 9999
worker_connections = []

# Define K-Fold value here, or dynamically based on configuration
K_FOLDS = 5  # Number of folds for K-Fold validation

def handle_worker_connection(conn, addr):
    logging.info(f"Worker connected from {addr}")
    config_path = "/app/data/distributed/configurations/wine"  # Path to configuration
    
    for fold_index in range(K_FOLDS):
        try:
            # Send configuration path and fold index to worker
            task = f"{config_path}|{fold_index}"
            conn.sendall(task.encode())
            logging.info(f"Sent task {task} to worker {addr}")
            
            # Wait for acknowledgment from worker
            response = conn.recv(1024).decode()
            if response == "ACK":
                logging.info(f"Worker {addr} completed fold {fold_index}")
        except socket.error as e:
            logging.error(f"Connection error with worker {addr}: {e}")
            break

    # Send shutdown signal after all folds are completed
    conn.sendall(b"shutdown")
    conn.close()

def start_coordinator():
    logging.info("Starting coordinator...")
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind(('0.0.0.0', COORDINATOR_PORT))
        server_socket.listen()
        logging.info(f"Coordinator listening on port {COORDINATOR_PORT}")
        
        while True:
            conn, addr = server_socket.accept()
            thread = threading.Thread(target=handle_worker_connection, args=(conn, addr))
            thread.start()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    start_coordinator()
