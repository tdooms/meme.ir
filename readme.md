# Information retrieval project
This project is made by Thomas Dooms and Basil Rommens

# How to run?
1. execute the build.py script to fetch and clean the main meme dataset
2. execute any other script (for example server.py to start up a backend)

The most useful files are classification and server.
Classification is a script that trains a model and saves it to a file along with other functions to analyse the model.
The server is a simple backend with a simple endpoint /generate/<text> that generates a meme based on the text.
OCR can be run from the `ocr.py` file (some values need to be changed). 
The other files in the `ocr` directory are used by the `ocr.py` file.
To generate the top 100 memes run the `top100.py` file, if some image fails 
remove it from that set.

# Data
## memes (the main database)
| Name   | Type      | Description                                                 |
|--------|-----------|-------------------------------------------------------------|
| url    | str       | url to the real meme image                                  |
| name   | str       | name, class of the meme                                     |
| post   | str       | url to the post of the meme                                 |
| views  | i32       | amount fo views                                             |
| votes  | i32       | amount of votes                                             |
| title  | str       | title of the post, defaults to name, do not use to classify |
| author | str       | author                                                      |
| boxes  | list[str] | captions of the meme                                        |

## templates
| Name      | Type |
|-----------|------|
| title     | str  |
| url       | str  |
| alt_names | str  |
| id        | i32  |

## statistics
| Name      | Type |
|-----------|------|
| path      | str  |
| count     | str  |
