import time
import numpy as np

class RCA_MetricsTracker:
    def __init__(self):
        # Dicionário para armazenar os tempos de cada incidente/anomalia
        self.incidents = {}

    def start_injection(self, incident_id):
        """T0: Momento em que o simulador envia o log anômalo."""
        self.incidents[incident_id] = {
            't0': time.time(), 
            't1': None, 
            't2': None
        }

    def mark_detected(self, incident_id):
        """T1: Momento em que a clusterização acusa a anomalia."""
        if incident_id in self.incidents:
            self.incidents[incident_id]['t1'] = time.time()

    def mark_isolated(self, incident_id):
        """T2: Momento em que a correlação de grafos aponta a causa raiz."""
        if incident_id in self.incidents:
            self.incidents[incident_id]['t2'] = time.time()

    def calculate_results(self):
        """Calcula o MTTD e MTTI finais em segundos."""
        mttd_list = []
        mtti_list = []

        for uid, times in self.incidents.items():
            if times['t1'] is not None and times['t0'] is not None:
                mttd_list.append(times['t1'] - times['t0'])
            
            if times['t2'] is not None and times['t1'] is not None:
                mtti_list.append(times['t2'] - times['t1'])

        mttd = float(np.mean(mttd_list)) if mttd_list else 0.0
        mtti =  float(np.mean(mtti_list)) if mtti_list else 0.0

        return {
            "Total_Incidentes": len(self.incidents),
            "MTTD_Segundos": round(mttd, 4),
            "MTTI_Segundos": round(mtti, 4)
        }