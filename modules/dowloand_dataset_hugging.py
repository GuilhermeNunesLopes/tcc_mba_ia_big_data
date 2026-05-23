import os
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

def download_dataset_file(repo_id: str, filename: str, local_dir: str = "logs_hugging") -> str:
    """
    Baixa um arquivo específico de um dataset no Hugging Face Hub.
    
    Certifique-se de ter o token configurado na variável de ambiente 'HF_TOKEN' 
    ou de ter rodado 'huggingface-cli login' no terminal para repositórios privados.
    """
    try:
        # Baixa o arquivo; o Hugging Face cuida do cache automaticamente
        file_path = hf_hub_download(
            repo_id=repo_id, 
            filename=filename, 
            repo_type="dataset", 
            local_dir=local_dir
        )
        
        print(f"✅ Arquivo baixado com sucesso em: {file_path}")
        
        # Mostra o tamanho do arquivo em Megabytes (MB) para melhor legibilidade
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"📊 Tamanho do arquivo: {file_size_mb:.2f} MB")
        
        return file_path
        
    except HfHubHTTPError as e:
        print(f"❌ Erro HTTP ao baixar o arquivo (verifique repo_id, filename ou suas permissões):\n{e}")
    except OSError as e:
        print(f"❌ Erro de sistema/disco ao tentar salvar o arquivo:\n{e}")
    except Exception as e:
        print(f"❌ Ocorreu um erro inesperado:\n{e}")
        
    return ""