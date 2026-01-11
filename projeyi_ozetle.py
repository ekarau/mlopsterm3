import os

# Hariç tutulacak klasörler ve dosya uzantıları (Gereksiz kalabalığı önlemek için)
IGNORE_DIRS = {'.git', 'venv', '__pycache__', '.idea', '.vscode', 'mlops_env', 'data', 'models'}
IGNORE_EXTENSIONS = {'.pyc', '.pkl', '.csv', '.png', '.jpg', '.jpeg', '.zip', '.gz'}

def save_project_structure_and_content(output_file="PROJE_OZETI.txt"):
    with open(output_file, "w", encoding="utf-8") as f:
        # 1. Önce Klasör Ağacını Yazalım
        f.write("=== PROJE KLASÖR YAPISI ===\n")
        for root, dirs, files in os.walk("."):
            # Gereksiz klasörleri atla
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            
            level = root.replace(".", "").count(os.sep)
            indent = " " * 4 * (level)
            f.write(f"{indent}{os.path.basename(root)}/\n")
            subindent = " " * 4 * (level + 1)
            for file in files:
                if not any(file.endswith(ext) for ext in IGNORE_EXTENSIONS):
                    f.write(f"{subindent}{file}\n")
        
        f.write("\n\n" + "="*50 + "\n\n")

        # 2. Şimdi Kod Dosyalarının İçeriğini Yazalım
        f.write("=== DOSYA İÇERİKLERİ ===\n")
        for root, dirs, files in os.walk("."):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            
            for file in files:
                # Sadece kod dosyalarını al (py, yml, yaml, txt, Dockerfile)
                if file.endswith(('.py', '.yml', '.yaml', 'Dockerfile', 'requirements.txt', '.md', '.sh')):
                    file_path = os.path.join(root, file)
                    f.write(f"\n{'='*20} {file_path} {'='*20}\n")
                    try:
                        with open(file_path, "r", encoding="utf-8") as code_file:
                            f.write(code_file.read())
                    except Exception as e:
                        f.write(f"--- Okunamadı: {e} ---\n")
                    f.write(f"\n{'='*50}\n")

    print(f"Tamamlandı! Tüm proje '{output_file}' dosyasına kaydedildi.")

if __name__ == "__main__":
    save_project_structure_and_content()