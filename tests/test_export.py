import io

from fedenet.export import export_onnx
from fedenet.models import FedeNetTiny


def test_export_onnx_writes_bytes(tmp_path):
    model = FedeNetTiny(in_ch=7).eval()
    path = tmp_path / "fedenet_tiny.onnx"
    export_onnx(model, out_path=str(path), opset=17)
    assert path.exists() and path.stat().st_size > 0


def test_export_onnx_to_bytesio():
    model = FedeNetTiny(in_ch=7).eval()
    buf = io.BytesIO()
    export_onnx(model, out_path=buf, opset=17)
    assert buf.getbuffer().nbytes > 0
