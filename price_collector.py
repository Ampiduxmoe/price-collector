from lost_ark import AucParser


def main():
    auc_parser = AucParser(tesseract_cmd=r'I:\Tesseract-OCR\tesseract')
    auc_parser.parse_auc_screenshots_folder(
        folder=r'latest screenshots', 
        extension=r'jpg', 
        output_folder=r'json dumps'
    )
    

if __name__ == '__main__':
    main()

