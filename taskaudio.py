import os
import stat
import shutil
import imageio_ffmpeg
from transformers import pipeline

ffmpeg_original = imageio_ffmpeg.get_ffmpeg_exe()
ffmpeg_dir = os.path.dirname(ffmpeg_original)
expected_path = os.path.join(ffmpeg_dir, "ffmpeg.exe")

if not os.path.exists(expected_path):
    shutil.copy2(ffmpeg_original, expected_path)
    os.chmod(expected_path, os.stat(expected_path).st_mode | stat.S_IEXEC)

os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]

asr = pipeline(
    "automatic-speech-recognition",
    model="./models/whisper",
    device=-1,
)

result = asr("audio.wav")
print(result["text"])