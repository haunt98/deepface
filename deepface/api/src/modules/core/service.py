# built-in dependencies
import traceback
from typing import Optional

# project dependencies
from deepface import DeepFace

# pylint: disable=broad-except


def represent(
    img_path: str,
    model_name: str,
    detector_backend: str,
    enforce_detection: bool,
    align: bool,
    anti_spoofing: bool,
    max_faces: Optional[int] = None,
):
    try:
        result = {}
        embedding_objs = DeepFace.represent(
            img_path=img_path,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
            anti_spoofing=anti_spoofing,
            max_faces=max_faces,
        )
        result["results"] = embedding_objs
        return result
    except Exception as err:
        tb_str = traceback.format_exc()
        return {"error": f"Exception while representing: {str(err)} - {tb_str}"}, 400


def verify(
    img1_path: str,
    img2_path: str,
    model_name: str,
    detector_backend: str,
    distance_metric: str,
    enforce_detection: bool,
    align: bool,
    anti_spoofing: bool,
):
    try:
        obj = DeepFace.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            model_name=model_name,
            detector_backend=detector_backend,
            distance_metric=distance_metric,
            align=align,
            enforce_detection=enforce_detection,
            anti_spoofing=anti_spoofing,
        )
        return obj
    except ValueError as verr:
        errRsp = {
            "error": f"Exception while verifying: {str(verr)}",
            "model_name": model_name,
            "detector_backend": detector_backend,
            "distance_metric": distance_metric,
        }
        if str(verr).startswith("Face could not be detected in img1_path"):
            errRsp["error_vi"] = "Không phát hiện được khuôn mặt trong ảnh 1"
            return errRsp, 200
        elif str(verr).startswith("Face could not be detected in img2_path"):
            errRsp["error_vi"] = "Không phát hiện được khuôn mặt trong ảnh 2"
            return errRsp, 200
        elif str(verr).startswith("Multiple faces are detected in img1_path"):
            errRsp["error_vi"] = "Phát hiện nhiều khuôn mặt trong ảnh 1"
            return errRsp, 200
        elif str(verr).startswith("Multiple faces are detected in img2_path"):
            errRsp["error_vi"] = "Phát hiện nhiều khuôn mặt trong ảnh 2"
            return errRsp, 200
        elif str(verr).startswith("Spoof detected in img1_path"):
            errRsp["error_vi"] = "Phát hiện ảnh giả mạo trong ảnh 1"
            return errRsp, 200
        elif str(verr).startswith("Spoof detected in img2_path"):
            errRsp["error_vi"] = "Phát hiện ảnh giả mạo trong ảnh 2"
            return errRsp, 200

        tb_str = traceback.format_exc()
        errRsp["traceback"] = tb_str
        return errRsp, 400
    except Exception as err:
        tb_str = traceback.format_exc()
        return {"error": f"Exception while verifying: {str(err)} - {tb_str}"}, 400


def analyze(
    img_path: str,
    actions: list,
    detector_backend: str,
    enforce_detection: bool,
    align: bool,
    anti_spoofing: bool,
):
    try:
        result = {}
        demographies = DeepFace.analyze(
            img_path=img_path,
            actions=actions,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
            silent=True,
            anti_spoofing=anti_spoofing,
        )
        result["results"] = demographies
        return result
    except ValueError as verr:
        errRsp = {
            "error": f"Exception while verifying: {str(verr)}",
            "detector_backend": detector_backend,
        }
        if str(verr).startswith("Face could not be detected"):
            errRsp["error_vi"] = "Không phát hiện được khuôn mặt"
            return errRsp, 200
        elif str(verr).startswith("Multiple faces are detected"):
            errRsp["error_vi"] = "Phát hiện nhiều khuôn mặt"
            return errRsp, 200
        elif str(verr).startswith("Spoof detected in the given image"):
            errRsp["error_vi"] = "Phát hiện ảnh giả mạo"
            return errRsp, 200

        tb_str = traceback.format_exc()
        errRsp["traceback"] = tb_str
        return errRsp, 400
    except Exception as err:
        tb_str = traceback.format_exc()
        return {"error": f"Exception while analyzing: {str(err)} - {tb_str}"}, 400
