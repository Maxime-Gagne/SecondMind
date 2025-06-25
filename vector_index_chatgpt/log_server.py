# -*- coding: utf-8 -*-
"""
Serveur de monitoring et logs pour SecondMind RAG
Version corrigée avec chemins absolus et améliorations
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template_string
import threading
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import psutil

# Configuration des chemins absolus
BASE_DIR = r"C:\Users\rag_personnel\Logs"
LOG_FILES = {
    'gradio_online': os.path.join(BASE_DIR, 'gradio_online.log'),
    'gradio_local': os.path.join(BASE_DIR, 'gradio_local.log'),
    'vectorize_online': os.path.join(BASE_DIR, 'vectorize_online.log'),
    'vectorize_local': os.path.join(BASE_DIR, 'vectorize_local.log'),
    'fix_faiss': os.path.join(BASE_DIR, 'fix_faiss_index.log'),
    'server': os.path.join(BASE_DIR, 'log_server.log')
}

INDEX_DIR = os.path.join(BASE_DIR, "vector_index_chatgpt")
CONVERSATIONS_FILE = os.path.join(BASE_DIR, "conversations_extraites.txt")

# Configuration Flask
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILES['server'], encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Variables globales pour le monitoring
log_data = {}
system_stats = {}
file_changes = []

class LogFileHandler(FileSystemEventHandler):
    """Gestionnaire des changements de fichiers de logs"""
    
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('.log'):
            file_changes.append({
                'file': os.path.basename(event.src_path),
                'timestamp': datetime.now().isoformat(),
                'type': 'modified',
                'path': event.src_path
            })
            # Garder seulement les 100 derniers changements
            if len(file_changes) > 100:
                file_changes.pop(0)

def read_log_file(filepath, lines=50):
    """Lit les dernières lignes d'un fichier de log"""
    try:
        if not os.path.exists(filepath):
            return []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
            return [line.strip() for line in all_lines[-lines:]]
    except Exception as e:
        logging.error(f"Erreur lecture log {filepath}: {e}")
        return [f"Erreur: {str(e)}"]

def get_system_info():
    """Récupère les informations système"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('C:')
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used_gb': memory.used / (1024**3),
            'memory_total_gb': memory.total / (1024**3),
            'disk_percent': disk.percent,
            'disk_used_gb': disk.used / (1024**3),
            'disk_total_gb': disk.total / (1024**3),
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logging.error(f"Erreur récupération système: {e}")
        return {}

def get_file_stats():
    """Récupère les statistiques des fichiers"""
    stats = {}
    
    files_to_check = {
        'conversations': CONVERSATIONS_FILE,
        'index_pkl': os.path.join(INDEX_DIR, 'index.pkl'),
        'index_faiss': os.path.join(INDEX_DIR, 'index.faiss')
    }
    
    for name, path in files_to_check.items():
        try:
            if os.path.exists(path):
                stat = os.stat(path)
                stats[name] = {
                    'exists': True,
                    'size_mb': stat.st_size / (1024*1024),
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                }
            else:
                stats[name] = {'exists': False}
        except Exception as e:
            stats[name] = {'error': str(e)}
    
    return stats

def update_log_data():
    """Met à jour les données de logs"""
    global log_data
    
    for name, filepath in LOG_FILES.items():
        log_data[name] = {
            'lines': read_log_file(filepath),
            'last_update': datetime.now().isoformat(),
            'exists': os.path.exists(filepath)
        }

def monitor_loop():
    """Boucle de monitoring en arrière-plan"""
    global system_stats
    
    while True:
        try:
            system_stats = get_system_info()
            update_log_data()
            time.sleep(10)  # Mise à jour toutes les 10 secondes
        except Exception as e:
            logging.error(f"Erreur monitoring: {e}")
            time.sleep(30)

# Routes Flask
@app.route('/')
def dashboard():
    """Page d'accueil du dashboard"""
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/logs')
def get_logs():
    """API pour récupérer les logs"""
    try:
        lines = int(request.args.get('lines', 50))
        log_type = request.args.get('type', 'all')
        
        if log_type == 'all':
            return jsonify(log_data)
        elif log_type in LOG_FILES:
            return jsonify({
                log_type: {
                    'lines': read_log_file(LOG_FILES[log_type], lines),
                    'last_update': datetime.now().isoformat()
                }
            })
        else:
            return jsonify({'error': 'Type de log invalide'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/system')
def get_system():
    """API pour récupérer les stats système"""
    try:
        return jsonify({
            'system': system_stats,
            'files': get_file_stats(),
            'changes': file_changes[-20:]  # 20 derniers changements
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health_check():
    """Vérification de santé du système"""
    try:
        health = {
            'status': 'ok',
            'timestamp': datetime.now().isoformat(),
            'services': {}
        }
        
        # Vérification des fichiers critiques
        for name, path in LOG_FILES.items():
            health['services'][name] = {
                'status': 'ok' if os.path.exists(path) else 'missing',
                'path': path
            }
        
        # Vérification des processus (ports)
        ports_to_check = [7860, 7861]  # Ports Gradio
        for port in ports_to_check:
            try:
                import socket
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    result = s.connect_ex(('127.0.0.1', port))
                    health['services'][f'port_{port}'] = {
                        'status': 'active' if result == 0 else 'inactive'
                    }
            except:
                health['services'][f'port_{port}'] = {'status': 'error'}
        
        return jsonify(health)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

# Template HTML pour le dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SecondMind RAG - Monitoring Dashboard</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(45deg, #2c3e50, #3498db);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            border-left: 4px solid #3498db;
        }
        .stat-card h3 {
            margin: 0 0 10px 0;
            color: #2c3e50;
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
        }
        .logs-section {
            padding: 30px;
        }
        .log-container {
            background: #2c3e50;
            color: #ecf0f1;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            font-family: 'Courier New', monospace;
            max-height: 400px;
            overflow-y: auto;
        }
        .log-line {
            padding: 2px 0;
            border-bottom: 1px solid #34495e;
        }
        .log-line:hover {
            background: #34495e;
        }
        .controls {
            margin: 20px 0;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s;
        }
        .btn-primary {
            background: #3498db;
            color: white;
        }
        .btn-primary:hover {
            background: #2980b9;
        }
        .btn-success {
            background: #27ae60;
            color: white;
        }
        .btn-success:hover {
            background: #219a52;
        }
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 5px;
        }
        .status-ok { background: #27ae60; }
        .status-error { background: #e74c3c; }
        .status-warning { background: #f39c12; }
        .auto-refresh {
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 10px;
            border-radius: 5px;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="auto-refresh" id="refreshStatus">
        🔄 Auto-refresh: ON
    </div>
    
    <div class="container">
        <div class="header">
            <h1>🧠 SecondMind RAG</h1>
            <p>Monitoring Dashboard - Système de surveillance en temps réel</p>
        </div>
        
        <div class="stats-grid" id="statsGrid">
            <!-- Stats will be loaded here -->
        </div>
        
        <div class="logs-section">
            <h2>📊 Logs en temps réel</h2>
            
            <div class="controls">
                <button class="btn btn-primary" onclick="refreshLogs()">🔄 Actualiser</button>
                <button class="btn btn-success" onclick="toggleAutoRefresh()">⏸️ Auto-refresh</button>
                <select id="logType" onchange="refreshLogs()">
                    <option value="all">Tous les logs</option>
                    <option value="gradio_online">Gradio Online</option>
                    <option value="gradio_local">Gradio Local</option>
                    <option value="vectorize_online">Vectorize Online</option>
                    <option value="vectorize_local">Vectorize Local</option>
                    <option value="fix_faiss">Fix FAISS</option>
                    <option value="server">Log Server</option>
                </select>
                <input type="number" id="logLines" value="50" min="10" max="500" placeholder="Lignes">
            </div>
            
            <div id="logsContainer">
                <!-- Logs will be loaded here -->
            </div>
        </div>
    </div>

    <script>
        let autoRefresh = true;
        let refreshInterval;

        function updateStats() {
            fetch('/api/system')
                .then(response => response.json())
                .then(data => {
                    const statsGrid = document.getElementById('statsGrid');
                    const system = data.system || {};
                    const files = data.files || {};
                    
                    statsGrid.innerHTML = `
                        <div class="stat-card">
                            <h3>💻 CPU</h3>
                            <div class="stat-value">${(system.cpu_percent || 0).toFixed(1)}%</div>
                        </div>
                        <div class="stat-card">
                            <h3>🧠 Mémoire</h3>
                            <div class="stat-value">${(system.memory_percent || 0).toFixed(1)}%</div>
                            <small>${(system.memory_used_gb || 0).toFixed(1)}GB / ${(system.memory_total_gb || 0).toFixed(1)}GB</small>
                        </div>
                        <div class="stat-card">
                            <h3>💾 Disque</h3>
                            <div class="stat-value">${(system.disk_percent || 0).toFixed(1)}%</div>
                            <small>${(system.disk_used_gb || 0).toFixed(1)}GB / ${(system.disk_total_gb || 0).toFixed(1)}GB</small>
                        </div>
                        <div class="stat-card">
                            <h3>📁 Fichiers</h3>
                            <div>
                                <span class="status-indicator ${files.conversations?.exists ? 'status-ok' : 'status-error'}"></span>
                                Conversations: ${files.conversations?.exists ? '✅' : '❌'}<br>
                                <span class="status-indicator ${files.index_pkl?.exists ? 'status-ok' : 'status-error'}"></span>
                                Index PKL: ${files.index_pkl?.exists ? '✅' : '❌'}<br>
                                <span class="status-indicator ${files.index_faiss?.exists ? 'status-ok' : 'status-error'}"></span>
                                Index FAISS: ${files.index_faiss?.exists ? '✅' : '❌'}
                            </div>
                        </div>
                    `;
                })
                .catch(error => {
                    console.error('Erreur lors du chargement des stats:', error);
                });
        }

        function refreshLogs() {
            const logType = document.getElementById('logType').value;
            const lines = document.getElementById('logLines').value;
            
            fetch(`/api/logs?type=${logType}&lines=${lines}`)
                .then(response => response.json())
                .then(data => {
                    const container = document.getElementById('logsContainer');
                    container.innerHTML = '';
                    
                    for (const [type, logData] of Object.entries(data)) {
                        if (logData.lines) {
                            const logDiv = document.createElement('div');
                            logDiv.innerHTML = `
                                <h3>📄 ${type.toUpperCase()} 
                                    <span class="status-indicator ${logData.exists ? 'status-ok' : 'status-error'}"></span>
                                </h3>
                                <div class="log-container">
                                    ${logData.lines.map(line => 
                                        `<div class="log-line">${line}</div>`
                                    ).join('')}
                                </div>
                            `;
                            container.appendChild(logDiv);
                        }
                    }
                })
                .catch(error => {
                    console.error('Erreur lors du chargement des logs:', error);
                    document.getElementById('logsContainer').innerHTML = 
                        '<div class="stat-card">❌ Erreur lors du chargement des logs</div>';
                });
        }

        function toggleAutoRefresh() {
            autoRefresh = !autoRefresh;
            const btn = event.target;
            const status = document.getElementById('refreshStatus');
            
            if (autoRefresh) {
                btn.textContent = '⏸️ Auto-refresh';
                status.textContent = '🔄 Auto-refresh: ON';
                startAutoRefresh();
            } else {
                btn.textContent = '▶️ Auto-refresh';
                status.textContent = '⏸️ Auto-refresh: OFF';
                clearInterval(refreshInterval);
            }
        }

        function startAutoRefresh() {
            refreshInterval = setInterval(() => {
                if (autoRefresh) {
                    updateStats();
                    refreshLogs();
                }
            }, 5000); // Refresh toutes les 5 secondes
        }

        // Initialisation
        document.addEventListener('DOMContentLoaded', function() {
            updateStats();
            refreshLogs();
            startAutoRefresh();
        });
    </script>
</body>
</html>
"""

def start_file_monitoring():
    """Démarre la surveillance des fichiers"""
    try:
        event_handler = LogFileHandler()
        observer = Observer()
        observer.schedule(event_handler, BASE_DIR, recursive=True)
        observer.start()
        logging.info("📁 Surveillance des fichiers démarrée")
        return observer
    except Exception as e:
        logging.error(f"Erreur surveillance fichiers: {e}")
        return None

if __name__ == "__main__":
    try:
        print("🚀 Démarrage du serveur de monitoring SecondMind RAG")
        print(f"📁 Répertoire surveillé: {BASE_DIR}")
        
        # Création des dossiers si nécessaire
        os.makedirs(BASE_DIR, exist_ok=True)
        os.makedirs(INDEX_DIR, exist_ok=True)
        
        # Démarrage de la surveillance des fichiers
        observer = start_file_monitoring()
        
        # Démarrage du thread de monitoring
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        
        print("🌐 Serveur web disponible sur: http://127.0.0.1:5000")
        print("📊 Dashboard de monitoring accessible")
        print("🔄 Mise à jour automatique toutes les 10 secondes")
        
        # Démarrage du serveur Flask
        app.run(
            host='127.0.0.1',
            port=5000,
            debug=False,
            threaded=True
        )
        
    except KeyboardInterrupt:
        print("\n👋 Arrêt demandé par l'utilisateur")
        if 'observer' in locals() and observer:
            observer.stop()
            observer.join()
    except Exception as e:
        print(f"❌ Erreur critique: {e}")
        logging.error(f"Erreur critique: {e}")
        input("Appuyez sur Entrée pour fermer...")