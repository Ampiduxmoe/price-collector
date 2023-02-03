from cgitb import text
import pytesseract
from pytesseract import Output
import cv2
import re
import json
import glob
from datetime import datetime


REFERENCE_BOUNDS = {
    'AUC_LABEL': (1232, 143, 1328, 165),
    'AVG_PRICE_LABEL': (1254, 370, 1388, 390),
    'LATEST_BUY_LABEL': (1481, 370, 1602, 389),
    'REMAININGS_LABEL': (1957, 370, 2086, 385),
    'AUC_WINDOW': (358, 128, 2192, 1274),
}

REFERENCE_POINTS = {
    'ITEM_NAME': (452, 278),
}

REFERENCE_DIMENSIONS = {
    'ITEM_HEIGHT': 76,
    'ITEM_NAME_WIDTH': 398,
    'ITEM_AVG_PRICE_WIDTH': 227,
    'ITEM_LATEST_BUY_WIDTH': 216,
    'ITEM_BEST_PRICE_WIDTH': 214,
    'ITEM_REMAININGS_WIDTH': 314,
    'COIN_RIGHT_OFFSET': 44,
    'PRICES_LEFT_OFFSET': 60,
    'PRICES_VERTICAL_OFFSET': 20,
    'REMAININGS_LEFT_OFFSET': 140,
    'ITEM_NAME_RIGHT_OFFSET': 60,
    'VERTICAL_BAR_WIDTH': 4,
}

LANG = 'rus'

ITEMS_PER_PAGE = 10
CROP_OFFSET = (452, 278, 15, 100)

class AucParser(object):
    tesseract_cmd = None
    unwanted_substr_in_itemnames = None
    remove_unwanted_itemname_part = None

    def __init__(self, tesseract_cmd):
        assert(tesseract_cmd)
        self.tesseract_cmd = tesseract_cmd
        self.unwanted_substr_in_itemnames = re.compile(r"\[.*\]")
        self.remove_unwanted_itemname_part = lambda s: self.unwanted_substr_in_itemnames.sub('', s)
        # self.tesseract_cmd = r'I:\Tesseract-OCR\tesseract'


    def parse_auc_screenshots_folder(self, folder, extension, output_folder):
        items = []
        if not folder.endswith('/'):
            folder += '/'
        files = glob.glob(f'{folder}*')
        for filename in files:
            if not filename.endswith(f'.{extension}'):
                continue
            items += self.parse_auc_screenshot(filename)
        now = datetime.now()
        timestamp = datetime.strftime(now, '%Y-%m-%d %H-%M')
        self.items_to_file(items, f'{output_folder}/{timestamp}.json',)


    def items_to_file(self, items, output_filename):
        with open(output_filename, 'w', encoding='utf-8') as file:
            file.write(json.dumps(items, indent=2, ensure_ascii=False))


    def parse_auc_screenshot(self, screenshot_path):
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
        orig_img = cv2.imread(screenshot_path)
        img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
        img = cv2.threshold(img, thresh=0, maxval=255, type=cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)[1]
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        key_bounds = self.get_key_bounds(img)
        assert(
            key_bounds['AUC_LABEL'] or 
            key_bounds['AVG_PRICE_LABEL'] or 
            key_bounds['LATEST_BUY_LABEL'] or 
            key_bounds['REMAININGS_LABEL']
        )
        # print('auc found' if key_bounds['AUC_LABEL'] else 'n')
        # print('avg price found' if key_bounds['AVG_PRICE_LABEL'] else 'n')
        # print('latest price found' if key_bounds['LATEST_BUY_LABEL'] else 'n')
        # print('best price found' if key_bounds['REMAININGS_LABEL'] else 'n')
        key_offset = self.find_key_offset(key_bounds)
        print('key_offset', key_offset)
        cropped_img = self.crop_auc(orig_img, key_offset, CROP_OFFSET)
        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        cropped_img = cv2.threshold(cropped_img, thresh=0, maxval=255, type=cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)[1]
        cropped_img_copy = cropped_img.copy()

        horizontal_bounds = self.get_horizontal_bounds(CROP_OFFSET)
        item_y_bounds = self.get_item_y_bounds(CROP_OFFSET)

        items = []
        for y_bounds in item_y_bounds:
            item = {}
            headers = [
                'ITEM_NAME',
                'ITEM_AVG_PRICE',
                'ITEM_LATEST_BUY',
                'ITEM_BEST_PRICE',
                'ITEM_REMAININGS',
            ]
            for header in headers:
                (left, top, right, bottom) = (horizontal_bounds[header][0], y_bounds[0], horizontal_bounds[header][1], y_bounds[1])
                config = '--psm 6'
                relevant_part = cropped_img[top:bottom, left:right]
                text_data = None
                if header not in ['ITEM_NAME']:
                    top += REFERENCE_DIMENSIONS['PRICES_VERTICAL_OFFSET']
                    bottom -= REFERENCE_DIMENSIONS['PRICES_VERTICAL_OFFSET']
                    # config += ' -c tessedit_char_whitelist=0123456789.,'
                if header == 'ITEM_NAME':
                    right -= REFERENCE_DIMENSIONS['ITEM_NAME_RIGHT_OFFSET']
                    # relevant_part = cv2.resize(relevant_part, (0, 0), fx=4, fy=4)
                    # text_data = pytesseract.image_to_string(relevant_part, lang=LANG, config=config).strip()
                else:
                    relevant_part = cv2.resize(relevant_part, (0, 0), fx=16, fy=16)
                    config = '--psm 6 digits --oem 3'
                    # text_data = pytesseract.image_to_string(relevant_part, config=config).strip()

                text_data = pytesseract.image_to_string(relevant_part, lang=LANG, config=config).strip()
                text_data = text_data + (16 - len(text_data)) * ' ' if header == 'ITEM_NAME' and len(text_data) < 16 else text_data
                # print(text_data)
                if text_data == '3':
                    cv2.imshow('img', relevant_part)
                    cv2.waitKey(0)
                item[header] = {
                    'bounds': (left, top, right, bottom),
                    'img': relevant_part,
                    'text': text_data
                }
                cv2.rectangle(cropped_img_copy, (left, top), (right, bottom), (0, 255, 0), 1)
            items.append(item)

        item_to_string = lambda x: f"{x['ITEM_NAME']['text']}:\t{x['ITEM_AVG_PRICE']['text']}\t{x['ITEM_LATEST_BUY']['text']}\t{x['ITEM_BEST_PRICE']['text']}\t{x['ITEM_REMAININGS']['text']}"
        for item in items:
            print(item_to_string(item))
        cv2.imshow('img', cropped_img_copy)
        cv2.waitKey(0)
        output_items = []
        no_dots = lambda s: s.replace(',', '').replace('.', '').replace(' ', '')
        for item in items:
            item_name = None
            if txt_val := item['ITEM_NAME']['text'].strip():
                item_name = self.remove_unwanted_itemname_part(txt_val).strip()

            item_avg_price = None
            if txt_val := no_dots(item['ITEM_AVG_PRICE']['text']).strip():
                item_avg_price = float(txt_val) / 10

            item_latest_price = None
            if txt_val := no_dots(item['ITEM_LATEST_BUY']['text']).strip():
                item_latest_price = int(txt_val)
            
            item_best_price = None
            if txt_val := no_dots(item['ITEM_BEST_PRICE']['text']).strip():
                item_best_price = int(txt_val)
            
            item_remainings = None
            if txt_val := no_dots(item['ITEM_REMAININGS']['text']).strip():
                item_remainings = int(txt_val)

            output_item = {
                'name': item_name,
                'avg_price': item_avg_price,
                'latest_price': item_latest_price,
                'best_price': item_best_price,
                'remainings': item_remainings,
            }
            no_item = not (
                item_name or 
                item_avg_price or 
                item_latest_price or
                item_best_price or
                item_remainings
            )
            if no_item:
                continue
            output_items.append(output_item)
        # print()
        # self.test_reference_items(items)
        return output_items


    def test_reference_items(self, items):
        avg_prices = []
        latest_buys = []
        best_prices = []
        remainings = []
        no_dots = lambda s: s.replace(',', '').replace('.', '')
        for item in items:
            avg_prices.append(no_dots(item['ITEM_AVG_PRICE']['text']))
            latest_buys.append(no_dots(item['ITEM_LATEST_BUY']['text']))
            best_prices.append(no_dots(item['ITEM_BEST_PRICE']['text']))
            remainings.append(no_dots(item['ITEM_REMAININGS']['text']))
        true_values = [
            (19, 2, 1, 249), #1
            (10, 1, 1, 40591), #2
            (16, 1, 1, 3097), #3
            (10, 1, 1, 470899), #4
            (10, 1, 1, 626), #5
            (10, 1, 1, 768), #6
            (19, 2, 2, 8760), #7
            (30, 3, 3, 1598), #8
            (25, 3, 3, 145836), #9
            (24, 5, 3, 30), #10
        ]
        true_values_count = len(true_values) * 4
        incorrect_values_count = 0
        for i, item in enumerate(items):
            prev_count = incorrect_values_count
            if avg_prices[i] != str(true_values[i][0]):
                incorrect_values_count += 1
                print(f'\t!! item #{i+1} has INCORRECT avg ({avg_prices[i]})')
                cv2.imshow('img', item['ITEM_AVG_PRICE']['img'])
                cv2.waitKey(0)
            if latest_buys[i] != str(true_values[i][1]):
                incorrect_values_count += 1
                print(f'\t!! item #{i+1} has INCORRECT latest ({latest_buys[i]})')
                cv2.imshow('img', item['ITEM_LATEST_BUY']['img'])
                cv2.waitKey(0)
            if best_prices[i] != str(true_values[i][2]):
                incorrect_values_count += 1
                print(f'\t!! item #{i+1} has INCORRECT best ({best_prices[i]})')
                cv2.imshow('img', item['ITEM_BEST_PRICE']['img'])
                cv2.waitKey(0)
            if remainings[i] != str(true_values[i][3]):
                incorrect_values_count += 1
                print(f'\t!! item #{i+1} has INCORRECT remaining ({remainings[i]})')
                cv2.imshow('img', item['ITEM_REMAININGS']['img'])
                cv2.waitKey(0)
            if prev_count != incorrect_values_count:
                print()
        print(f'  passed {true_values_count-incorrect_values_count}/{true_values_count}')


    def get_key_bounds(self, img):
        img_dict = pytesseract.image_to_data(img, lang=LANG, output_type=Output.DICT)
        key_bounds = {
            'AUC_LABEL': self.find_phrase_rects(
                img, 'Аукцион', img_dict=img_dict, max_words=1, exact_match=False
            ),
            'AVG_PRICE_LABEL': self.find_phrase_rects(
                img, 'Средняя цена', img_dict=img_dict, max_words=2, exact_match=False
            ),
            'LATEST_BUY_LABEL': self.find_phrase_rects(
                img, 'Актуал. цена', img_dict=img_dict, max_words=2, exact_match=False
            ),
            'REMAININGS_LABEL': self.find_phrase_rects(
                img, 'Мин. остаток', img_dict=img_dict, max_words=2, exact_match=False
            ),
        }
        return key_bounds


    def find_key_offset(self, key_bounds):
        key_names = [
            'AUC_LABEL', 
            'AVG_PRICE_LABEL', 
            'LATEST_BUY_LABEL', 
            'REMAININGS_LABEL',
        ]
        x_offsets = []
        y_offsets = []
        for name in key_names:
            bounds = key_bounds[name]
            if not bounds:
                continue
            if isinstance(bounds, list):
                if len(bounds) > 1:
                    continue
                bounds = bounds[0]
            
            reference_bounds = REFERENCE_BOUNDS[name]
            x_offset = (bounds[0] + bounds[2] - reference_bounds[0] - reference_bounds[2]) / 2.0
            y_offset = (bounds[1] + bounds[3] - reference_bounds[1] - reference_bounds[3]) / 2.0
            x_offsets.append(x_offset)
            y_offsets.append(y_offset)
        arr_avg = lambda x: float(sum(x)) / len(x)
        key_offset = (round(arr_avg(x_offsets)), round(arr_avg(y_offsets)))
        return key_offset


    def crop_auc(self, img, key_offset, crop_offset):
        left_crop = REFERENCE_BOUNDS['AUC_WINDOW'][0] + key_offset[0] + crop_offset[0]
        top_crop = REFERENCE_BOUNDS['AUC_WINDOW'][1] + key_offset[1] + crop_offset[1]
        right_crop = REFERENCE_BOUNDS['AUC_WINDOW'][2] + key_offset[0] - crop_offset[2]
        bottom_crop = REFERENCE_BOUNDS['AUC_WINDOW'][3] + key_offset[1] - crop_offset[3]
        
        img_copy = img.copy()
        cv2.rectangle(img_copy, (left_crop, top_crop), (right_crop, bottom_crop), (0, 255, 0), 2)
        cv2.imshow('img', img_copy)
        cv2.waitKey(0)

        cropped_img = img[top_crop:bottom_crop, left_crop:right_crop]
        return cropped_img


    def get_horizontal_bounds(self, crop_offset):
        horizontal_bounds = {
            'ITEM_NAME': (
                REFERENCE_POINTS['ITEM_NAME'][0] - crop_offset[0],
                REFERENCE_POINTS['ITEM_NAME'][0] + REFERENCE_DIMENSIONS['ITEM_NAME_WIDTH'] - crop_offset[0]
            )
        }
        horizontal_bounds['ITEM_AVG_PRICE'] = (
            horizontal_bounds['ITEM_NAME'][1],
            horizontal_bounds['ITEM_NAME'][1] + REFERENCE_DIMENSIONS['ITEM_AVG_PRICE_WIDTH']
        )
        horizontal_bounds['ITEM_LATEST_BUY'] = (
            horizontal_bounds['ITEM_AVG_PRICE'][1],
            horizontal_bounds['ITEM_AVG_PRICE'][1] + REFERENCE_DIMENSIONS['ITEM_LATEST_BUY_WIDTH']
        )
        horizontal_bounds['ITEM_BEST_PRICE'] = (
            horizontal_bounds['ITEM_LATEST_BUY'][1],
            horizontal_bounds['ITEM_LATEST_BUY'][1] + REFERENCE_DIMENSIONS['ITEM_BEST_PRICE_WIDTH']
        )
        horizontal_bounds['ITEM_REMAININGS'] = (
            horizontal_bounds['ITEM_BEST_PRICE'][1] + REFERENCE_DIMENSIONS['REMAININGS_LEFT_OFFSET'],
            horizontal_bounds['ITEM_BEST_PRICE'][1] + REFERENCE_DIMENSIONS['ITEM_REMAININGS_WIDTH']
        )
        for name in horizontal_bounds:
            horizontal_bounds[name] = (
                horizontal_bounds[name][0],
                horizontal_bounds[name][1] - REFERENCE_DIMENSIONS['VERTICAL_BAR_WIDTH']
            )
        for name in horizontal_bounds:
            if name in ['ITEM_NAME', 'ITEM_REMAININGS']:
                continue
            horizontal_bounds[name] = (
                horizontal_bounds[name][0] + REFERENCE_DIMENSIONS['PRICES_LEFT_OFFSET'],
                horizontal_bounds[name][1] - REFERENCE_DIMENSIONS['COIN_RIGHT_OFFSET']
            )
        return horizontal_bounds


    def get_item_y_bounds(self, crop_offset):
        return [
            (
                REFERENCE_POINTS['ITEM_NAME'][1] + REFERENCE_DIMENSIONS['ITEM_HEIGHT'] * (i) - crop_offset[1], 
                REFERENCE_POINTS['ITEM_NAME'][1] + REFERENCE_DIMENSIONS['ITEM_HEIGHT'] * (i + 1) - crop_offset[1]
            ) for i in range(ITEMS_PER_PAGE)
        ]


    def draw_boxes(self, img, boxes):
        for bounds in boxes:
            if not bounds:
                continue
            (left, top, right, bottom) = bounds[0], bounds[1], bounds[2], bounds[3]
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 1)


    def find_phrase_rects(self, img, phrase, img_dict=None, max_words=1, exact_match=False):
        substr_in = lambda substr, str: substr == str
        if not exact_match:
            substr_in = lambda substr, str: substr.lower() in str.lower()

        if not img_dict:
            img_dict = pytesseract.image_to_data(img, lang='rus', output_type=Output.DICT)
        d_text = img_dict['text']

        found_indices = []
        for i, _ in enumerate(d_text):
            last_search_range_index = min(i + max_words, len(d_text))
            labels_in_range = [d_text[j] for j in range(i, last_search_range_index)]
            current_label = labels_in_range[0]
            if substr_in(phrase, current_label):
                found_indices.append(i)
            else:
                for j in range(1, len(labels_in_range)):
                    current_label = f'{current_label} {labels_in_range[j]}'
                    if substr_in(phrase, current_label):
                        found_indices.append([i, i+j])
                        break

        bounds = []
        for index in found_indices:
            if isinstance(index, list):
                start = index[0]
                end = index[1] + 1
                (xs, ys, ws, hs) = (
                    img_dict['left'][start:end], 
                    img_dict['top'][start:end], 
                    img_dict['width'][start:end], 
                    img_dict['height'][start:end]
                )
                left = min(xs)
                top = min(ys)
                right = max([x + ws[i] for i, x in enumerate(xs)])
                bottom = max([y + hs[i] for i, y in enumerate(ys)])
                bounds.append((left, top, right, bottom))
                continue
            i = index
            (x, y, w, h) = (img_dict['left'][i], img_dict['top'][i], img_dict['width'][i], img_dict['height'][i])
            bounds.append((x, y, x + w, y + h))
        return bounds
