from src.detect import find_epic_moment
from pathlib import Path


if __name__ == "__main__":
    test_folder = Path("audio_load")
    
    for track in test_folder.glob("*.wav"):
        find_epic_moment(str(track))
        print("-" * 30)




