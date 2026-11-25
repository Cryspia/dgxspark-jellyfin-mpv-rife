import unittest

from vs_gpu_helpers import rife_engine_metadata


class RifeMetadataTest(unittest.TestCase):
    def test_supported_model_layouts_without_cuda(self):
        self.assertEqual(rife_engine_metadata("4.26", 1080, 1920),
                         {"ph": 1088, "pw": 1920, "encode_channel": 4})
        self.assertEqual(rife_engine_metadata("4.26", 2160, 3840),
                         {"ph": 2176, "pw": 3840, "encode_channel": 4})
        self.assertEqual(rife_engine_metadata("4.6", 1080, 1920),
                         {"ph": 1088, "pw": 1920, "encode_channel": 0})

    def test_rejects_unknown_model_or_dimensions(self):
        for model, height, width, scale in (("5.0", 1080, 1920, 1),
                                            ("4.26", 0, 1920, 1),
                                            ("4.26", 1080, 1920, 0)):
            with self.assertRaises(ValueError):
                rife_engine_metadata(model, height, width, scale)
