import os

def modify_cls_in_folder(input_folder, output_folder):
    # 출력 폴더가 없으면 생성
    os.makedirs(output_folder, exist_ok=True)
    
    # 입력 폴더에서 모든 텍스트 파일 순차적으로 읽기
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_folder, filename)
            output_file_path = os.path.join(output_folder, filename)
            
            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            # 각 줄을 읽고 cls가 1인 경우 9로 수정
            modified_lines = []
            for line in lines:
                parts = line.strip().split()
                cls = int(parts[0])  # cls 값을 가져옴
                # if cls in [0, 1, 2, 3, 4, 5, 6]:
                    # if cls == 6:
                        # print(f"Modified file saved to: {output_file_path}")
                        # parts[0] = '6'  # cls가 1이면 9로 변경

                if cls in [0, 1, 2, 3, 5, 7]:
                    if cls == 5:
                        # print(f"Modified file saved to: {output_file_path}")
                        parts[0] = '4'  # cls가 1이면 9로 변경

                    elif cls == 7:
                        parts[0] = "5"

                    modified_lines.append(" ".join(parts))  # 수정된 줄을 리스트에 추가
            
            # 수정된 내용을 새로운 파일에 저장
            with open(output_file_path, 'w') as file:
                file.write("\n".join(modified_lines))

            print(f"Modified file saved to: {output_file_path}")



def remove_npy() -> None:
    import glob
    npy_files = glob.glob(os.path.join(os.getcwd(), "dataset", '**', '*.npy'), recursive=True)
    cache_files = glob.glob(os.path.join(os.getcwd(), "dataset", '**', '*.cache'), recursive=True)

    for file_path in npy_files:
        try:
            os.remove(file_path)
            # print(f"파일 '{file_path}'이(가) 삭제되었습니다.")
        except Exception as e:
            print(f"파일 '{file_path}'을(를) 삭제하는 중 오류가 발생했습니다: {e}")

    for file_path in cache_files:
        try:
            os.remove(file_path)
            # print(f"파일 '{file_path}'이(가) 삭제되었습니다.")
        except Exception as e:
            print(f"파일 '{file_path}'을(를) 삭제하는 중 오류가 발생했습니다: {e}")


# 사용 예시
input_folder = "./dataset/coco/val/labels"       # 텍스트 파일들이 있는 폴더 경로
output_folder = "./dataset/coco/val/labels_new"     # 수정된 파일을 저장할 폴더 경로
modify_cls_in_folder(input_folder, output_folder)

remove_npy()

os.system("chmod 777 -R ./")