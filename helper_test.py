from helper import *
import unittest


class HelperTest(unittest.TestCase):
    def test_get_evt_name_from_code(self):
        self.assertEqual(get_evt_name_from_code("S"), "Sent")
        self.assertEqual(get_evt_name_from_code("B"), "Soft bounce")
        self.assertEqual(get_evt_name_from_code("H"), "Hard bounce")
        self.assertEqual(get_evt_name_from_code("d"), "Deferred")
        self.assertEqual(get_evt_name_from_code("D"), "Delivered")
        self.assertEqual(get_evt_name_from_code("L"), "Loaded by proxy")
        self.assertEqual(get_evt_name_from_code("F"), "First opening")
        self.assertEqual(get_evt_name_from_code("O"), "Opened")
        self.assertEqual(get_evt_name_from_code("C"), "Clicked")
        self.assertEqual(get_evt_name_from_code("U"), "Unsubscribed")
        self.assertEqual(get_evt_name_from_code("A"), "Abuse complaint")

    def test_get_evt_idx(self):
        self.assertEqual(get_evt_idx("S"), 0)
        self.assertEqual(get_evt_idx("B"), 1)
        self.assertEqual(get_evt_idx("H"), 2)
        self.assertEqual(get_evt_idx("d"), 3)
        self.assertEqual(get_evt_idx("D"), 4)
        self.assertEqual(get_evt_idx("L"), 5)
        self.assertEqual(get_evt_idx("F"), 6)
        self.assertEqual(get_evt_idx("O"), 7)
        self.assertEqual(get_evt_idx("C"), 8)
        self.assertEqual(get_evt_idx("U"), 9)
        self.assertEqual(get_evt_idx("A"), 10)
