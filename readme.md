# Information retrieval project
This project is made by Thomas Dooms and Basil Rommens


# Data
## memes (the main database)
┌──────────────────┬─────────────────┬───────┬───────┬───────────┬───────────────┬─────────────────┐
│ url              ┆ post            ┆ views ┆ votes ┆ title     ┆ author        ┆ boxes           │
│ ---              ┆ ---             ┆ ---   ┆ ---   ┆ ---       ┆ ---           ┆ ---             │
│ str              ┆ str             ┆ i32   ┆ i32   ┆ str       ┆ str           ┆ list[str]       │
└──────────────────┴─────────────────┴───────┴───────┴───────────┴───────────────┴─────────────────┘

## templates
┌────────────────────────────┬────────────────────────────┬────────────────────────────┬───────────┐
│ title                      ┆ url                        ┆ alt_names                  ┆ id        │
│ ---                        ┆ ---                        ┆ ---                        ┆ ---       │
│ str                        ┆ str                        ┆ str                        ┆ str       │
└────────────────────────────┴────────────────────────────┴────────────────────────────┴───────────┘

## statistics
┌───────────────────────────────────────────────────┬──────────────────────────────────────────────┐
│ path                                              ┆ count                                        │
│ ---                                               ┆ ---                                          │
│ str                                               ┆ str                                          │
└───────────────────────────────────────────────────┴──────────────────────────────────────────────┘

# TODO
- [x] Find a database of memes
- [x] Create s script to clean the data into a usable format
- [.] Generate mean embeddings per template and plot with t-SNE
- [ ] Look at fine-tuning a model like [distilbert](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)
- [ ] Look at gathering more data (with Google search or from a subreddit)
- [ ] Look at using a model like [CLIP](https://github.com/openai/CLIP)

# Finalising
- [ ] Write the report
- [ ] Make the presentation
- [ ] Make a CLI
- [ ] Make a web UI
- [ ] Generate real memes instead of classifying into templates