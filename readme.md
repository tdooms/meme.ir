# Information retrieval project
This project is made by Thomas Dooms and Basil Rommens

# How to run?
1. execute the build.py script to fetch and clean the main meme dataset
2. execute any other script 


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

# TODO
- [x] Find a database of memes
- [x] Create s script to clean the data into a usable format
- [x] Generate mean embeddings per template and plot with t-SNE
- [ ] Look at fine-tuning a model like [distilbert](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)
- [ ] Look at gathering more data (with Google search or from a subreddit)
- [ ] Look at using a model like [CLIP](https://github.com/openai/CLIP)

# Finalising
- [ ] Write the report
- [ ] Make the presentation
- [ ] Make a CLI
- [ ] Make a web UI
- [ ] Generate real memes instead of classifying into templates