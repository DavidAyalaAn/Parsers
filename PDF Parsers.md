
# Pytesseract

This is a text parser from images that extracts additional attributes. This attributes can help to apply rules to identify the type of data.
This method is a good option for old pdfs that seems to be scanned versions of documents.

As we are working with images, first we need to generate the images for each pdf page.

```python
from pdf2image import convert_from_path

images = convert_from_path(
	pdf_path, 
	first_page=1,
	last_page=2, 
	dpi=300, 
	output_folder='./output_folder',
	output_file='prefix_name', 
	fmt="png"
)
```

For each image we extract the data.

```python
import pytesseract

image_files = os.listdir('.\source_folder')
ocr_data = pd.DataFrame()

for img_file in tqdm(image_files):
	#Load the image
	image = Image.open(os.path.join(source_folder,img_file))
	
	#Extracts data from image and saves to data frame
	image_arr = np.array(image)
	img_data = pytesseract.image_to_data(
		image_arr, 
		output_type=pytesseract.Output.DATAFRAME
	)

	#Adds the data from the image to our dataframe
	ocr_data = pd.concat([ocr_data,img_data])

```

The output obtained is a dataframe splitted by word with some page attributes.

| level | page_num | block_num | par_num | line_num | word_num | left | top  | width | height | conf      | text  |
| ----- | -------- | --------- | ------- | -------- | -------- | ---- | ---- | ----- | ------ | --------- | ----- |
| 5     | 1        | 4         | 1       | 1        | 1        | 202  | 1783 | 111   | 25     | 83.714577 | A-121 |

- level: the hierarchy level (1 = page, 2 = block, 3 = paragraph, 4 = line, 5 = word)
- page_num: page number
- block_num: block number within the page
- par_num: paragraph number within the block
- line_num: line number within the paragraph
- word_num: word number within the line
- text: the recognized text
- conf: confidence score



# Gemini Parser
The use of an llm give better results that traditional parsers.

The most relevant aspect while working with an llm is to build a proper prompt for our use case.

```python

prompt = '''
Identify all the terms and definitions in the uploaded file.
To make the distinction you considere:
    - The term is followed by its definition.
    - The term has a bold font and the definition has a normal font.
    - The term can be composed for more than one word.
    - The definition sometimes can start with a lowercase letter.
    - If you find quotes in the PDF use two quotes, example ""this is a quoted text"".
    - Only respond with the formated text.
    
Return the information in a csv format like this:
    "term1","definition1"
    "term2","definition2"
'''
```


To do the parsing we have to upload the file and also send our prompt with a code like this:

```python
import google.generativeai as genai

#API key setting
genai.configure(api_key='AI...')

#llm setup
model = genai.GenerativeModel(
    model_name='gemini-1.5-flash',
    generation_config=genai.types.GenerationConfig(
        temperature=0.0,
        max_output_tokens=None
    )
)

#uploads file to gemini llm
upload_file = genai.upload_file(path=os.path.join(path,file))

#Model invoke
response = model.generate_content(
        [prompt, upload_file]
    )
response.text

```