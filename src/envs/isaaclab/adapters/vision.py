import cv2
import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    F = None


class ImageAcquisitionError(RuntimeError):
    pass


def is_image_array(array) -> bool:
    if array is None:
        return False
    if torch is not None and torch.is_tensor(array):
        return array.ndim in (3, 4)
    arr = np.asarray(array)
    if arr.ndim == 4:
        return True
    if arr.ndim == 3:
        return True
    return False


def _to_numpy(value):
    if torch is not None and torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class IsaacLabImageProvider:
    def __init__(self, env, request, task_spec):
        self._env = env
        self._request = request
        self._task_spec = task_spec

    @property
    def image_shape(self):
        return self.payload_image_shape

    @property
    def payload_image_shape(self):
        if self._request.flatten_obs:
            return (self._request.render_size * self._request.render_size * 3,)
        return self.image_space_shape

    @property
    def image_space_shape(self):
        return (self._request.render_size, self._request.render_size, 3)

    def placeholder_image(self):
        return np.zeros(self.payload_image_shape, dtype=np.uint8)

    def placeholder_image_tensor(self):
        if torch is None:
            return None
        device = self._resolve_tensor_device(None)
        return torch.zeros(self.payload_image_shape, dtype=torch.uint8, device=device)

    def capture(self, obs_dict, extras, allow_placeholder: bool):
        image = None
        if self._request.image_source in ("auto", "camera"):
            image = self._extract_camera_image(obs_dict, extras)
            if image is not None:
                return self._finalize_image(image)
            if self._request.image_source == "camera" and self._request.encoder:
                raise ImageAcquisitionError(
                    f"Camera image requested for {self._task_spec.env_id}, but no camera observation was found."
                )

        if self._request.image_source in ("auto", "render") or not self._request.encoder:
            image = self._render_image()
            if image is not None:
                return self._finalize_image(image)
            if self._request.image_source == "render" and self._request.encoder:
                raise ImageAcquisitionError(
                    f"Render image requested for {self._task_spec.env_id}, but env.render() returned nothing."
                )

        if allow_placeholder:
            return self.placeholder_image()

        raise ImageAcquisitionError(
            f"Failed to acquire an image for {self._task_spec.env_id} "
            f"(source={self._request.image_source}, encoder={self._request.encoder})."
        )

    def capture_tensor(self, obs_dict, extras, allow_placeholder: bool, *, batched: bool = False):
        if torch is None:
            return None

        image = None
        if self._request.image_source in ("auto", "camera"):
            image = self._extract_camera_image(obs_dict, extras)
            if image is not None:
                return self._finalize_image_tensor(image, batched=batched)
            if self._request.image_source == "camera" and self._request.encoder:
                raise ImageAcquisitionError(
                    f"Camera image requested for {self._task_spec.env_id}, but no camera observation was found."
                )

        if self._request.image_source in ("auto", "render") or not self._request.encoder:
            image = self._render_image()
            if image is not None:
                return self._finalize_image_tensor(image, batched=batched)
            if self._request.image_source == "render" and self._request.encoder:
                raise ImageAcquisitionError(
                    f"Render image requested for {self._task_spec.env_id}, but env.render() returned nothing."
                )

        if allow_placeholder:
            return self.placeholder_image_tensor()

        raise ImageAcquisitionError(
            f"Failed to acquire an image tensor for {self._task_spec.env_id} "
            f"(source={self._request.image_source}, encoder={self._request.encoder})."
        )

    def _extract_camera_image(self, obs_dict, extras):
        keys = []
        if self._request.camera_key:
            keys.append(self._request.camera_key)
        if self._task_spec.camera_obs_key:
            keys.append(self._task_spec.camera_obs_key)
        keys.extend(
            [
                "front_rgb",
                "head_rgb",
                "rgb",
                "image",
                "pixels",
                "camera",
                "rgb_image",
                "camera_0",
                "left_wrist_rgb",
                "right_wrist_rgb",
            ]
        )

        for data in (obs_dict, extras):
            image = self._search_named_image(data, keys)
            if image is not None:
                return image
        return None

    def _search_named_image(self, value, keys):
        if isinstance(value, dict):
            for key in keys:
                if key in value and is_image_array(value[key]):
                    return value[key]
            for nested in value.values():
                found = self._search_named_image(nested, keys)
                if found is not None:
                    return found
        elif isinstance(value, (list, tuple)):
            for nested in value:
                found = self._search_named_image(nested, keys)
                if found is not None:
                    return found
        return None

    def _render_image(self):
        try:
            image = self._env.render()
        except TypeError:
            try:
                image = self._env.render(mode=self._request.render_mode)
            except Exception:
                return None
        except Exception:
            return None
        return image

    def _resolve_tensor_device(self, value):
        if torch is not None and torch.is_tensor(value):
            return value.device
        request_device = getattr(self._request, "device", None)
        if request_device not in (None, ""):
            return request_device
        env_device = getattr(getattr(self._env, "unwrapped", self._env), "device", None)
        if env_device not in (None, ""):
            return env_device
        return "cpu"

    def _to_tensor(self, value):
        if torch is None:
            raise RuntimeError("Torch is required for Isaac Lab tensor image capture.")
        if torch.is_tensor(value):
            return value
        return torch.as_tensor(np.asarray(value), device=self._resolve_tensor_device(value))

    def _to_hwc_tensor(self, value, *, batched: bool = False):
        tensor = self._to_tensor(value)
        if tensor.ndim == 4:
            if tensor.shape[-1] in (1, 3, 4):
                pass
            elif tensor.shape[1] in (1, 3, 4):
                tensor = tensor.movedim(1, -1)
            else:
                raise ImageAcquisitionError(f"Unsupported batched tensor image shape: {tuple(tensor.shape)}")
        elif tensor.ndim == 3:
            if tensor.shape[-1] in (1, 3, 4):
                pass
            elif tensor.shape[0] in (1, 3, 4):
                tensor = tensor.movedim(0, -1)
            else:
                raise ImageAcquisitionError(f"Unsupported tensor image shape: {tuple(tensor.shape)}")
        elif tensor.ndim == 2:
            tensor = tensor.unsqueeze(-1)
        else:
            raise ImageAcquisitionError(f"Unsupported tensor image shape: {tuple(tensor.shape)}")

        if tensor.ndim == 4 and not batched:
            if tensor.shape[0] == 1:
                tensor = tensor[0]
            else:
                tensor = tensor[0]
        return tensor

    def _finalize_image_tensor(self, value, *, batched: bool = False):
        tensor = self._to_hwc_tensor(value, batched=batched)

        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        if tensor.shape[-1] == 1:
            tensor = tensor.repeat_interleave(3, dim=-1)
        elif tensor.shape[-1] == 4:
            tensor = tensor[..., :3]

        if tensor.dtype != torch.uint8:
            tensor = tensor.to(dtype=torch.float32)
            max_value = float(tensor.max().item()) if tensor.numel() else 0.0
            if max_value <= 1.0:
                tensor = torch.clamp(tensor * 255.0, 0, 255)
            else:
                tensor = torch.clamp(tensor, 0, 255)
            tensor = tensor.to(dtype=torch.uint8)

        target_hw = (self._request.render_size, self._request.render_size)
        if tuple(tensor.shape[1:3]) != target_hw:
            if F is None:
                raise RuntimeError("Torch functional interpolate is required for GPU Isaac Lab image resizing.")
            tensor = tensor.movedim(-1, 1).to(dtype=torch.float32)
            tensor = F.interpolate(tensor, size=target_hw, mode="bilinear", align_corners=False)
            tensor = torch.clamp(tensor, 0, 255).to(dtype=torch.uint8).movedim(1, -1)

        if squeeze_batch:
            tensor = tensor[0]

        if self._request.flatten_obs:
            if tensor.ndim == 4:
                return tensor.reshape(tensor.shape[0], -1)
            return tensor.reshape(-1)
        return tensor

    def _finalize_image(self, value):
        if torch is not None and torch.is_tensor(value):
            return self._finalize_image_tensor(value).detach().cpu().numpy()

        array = _to_numpy(value)
        if array.ndim == 4:
            array = array[0]
        if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
            array = np.moveaxis(array, 0, -1)
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=-1)
        if array.ndim != 3:
            raise ImageAcquisitionError(f"Unsupported image shape: {tuple(array.shape)}")

        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        elif array.shape[-1] == 4:
            array = array[..., :3]

        if array.dtype != np.uint8:
            max_value = float(array.max()) if array.size else 0.0
            if max_value <= 1.0:
                array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
            else:
                array = np.clip(array, 0, 255).astype(np.uint8)

        if array.shape[:2] != (self._request.render_size, self._request.render_size):
            array = cv2.resize(array, (self._request.render_size, self._request.render_size), interpolation=cv2.INTER_LINEAR)

        if self._request.flatten_obs:
            return array.reshape(-1)
        return array
