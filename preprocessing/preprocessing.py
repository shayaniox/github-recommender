"""Stage C - topic normalization.

Raw GitHub topics are noisy: `python3`, `fast-api`, `restful-apis`, `js`, `to`. This module
rewrites them into a clean vocabulary with 13 rule-driven passes.

RUNNING EXAMPLE, used in the comments throughout. Row 1 of `repositories_test.csv`:

    repo    'freeCodeCamp/freeCodeCamp'
    topics  'learn-to-code,nonprofits,programming,nodejs,react,d3,careers,education,teachers,javascript'

After `preprocess()` that becomes, as a Python list:

    ['learn', 'nodejs', 'react', 'd3', 'education', 'teacher', 'javascript']

10 topics in, 7 out - see `preprocess()` for the pass-by-pass trace.
"""

import os
from nltk.corpus import wordnet as wn
from nltk.corpus import words as w
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

REWRITING_RULE_FOLDER = './preprocessing/rules'

# Only the first time
# import nltk
# nltk.download('words')
# nltk.download('wordnet')

stops = stopwords.words("english")
words = w.words()
wordswn = wn.words()
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def topics_string_to_list(df):
    """Final shaping: the comma-joined string becomes a de-duplicated list.

        topics  'learn,nodejs,react,d3,education,teacher,javascript'   (str)
             -> ['learn', 'nodejs', 'react', 'd3', 'education', 'teacher', 'javascript']   (list)

    `set` de-duplicates but discards order, so the list order is arbitrary from here on.
    Nothing downstream depends on it: the exporters only number the topics, and TopFilter
    treats each repo's topics as a set.
    """
    def make_topics(topics):
        result = set()
        for t in topics.split(','):
            if t.strip() == '':
                continue
            result.add(t)
        return (list(result))
    df.topics = df.topics.apply(make_topics)
    return df

def get_topic_frequency_map_topics_as_list(df):
    """How many repositories carry each topic. Expects `topics` already as lists.

        df.topics  [['python', 'api', 'docker'], ['python', 'rest-api'], ['python']]
                -> {'python': 3, 'api': 1, 'docker': 1, 'rest-api': 1}
    """
    def count_topics(topics):
        for t in topics:
            frequency_map[t] = frequency_map.get(t, 0) + 1

    frequency_map = {}
    df.topics.apply(count_topics)
    return frequency_map

def apply_csv(file_name, df):
    """The 1:N rewriting primitive. Nine of the 13 passes are this function.

    Rule files are headerless CSV: `frequency, target_topic, replacement1, replacement2, ...`
    Column 0 (the corpus frequency the rule was derived from) is read but never used.

        'split_dash_topics.csv'   line: `12316,machine-learning,machine,learning,,,,,,`
                                  -> edit_list['machine-learning'] = ['machine', 'learning']

    Two sentinel values sit in the replacement slot:

        `2390,to,-1`        -1  delete the topic          'to'      -> (gone)
        `14390,js,-2`       -2  keep it unchanged         'js'      -> 'js'
        `9386,python3,python`   otherwise expand 1 -> N   'python3' -> 'python'

    Worked on the running example, `split_dash_topics.csv`:

        in   'learn-to-code,nonprofits,programming,nodejs,react,d3,careers,education,teachers,javascript'
        out  'learn,to,code,nonprofits,programming,nodejs,react,d3,careers,education,teachers,javascript'

    Note the `set` on the return: replacements that collide are merged, and topic order is
    lost on every pass. A repo left with no topics at all is dropped from the frame.
    """
    edit_list = {}
    with open(file_name) as file:
        for line in file.read().strip('\n').split('\n'):
            items = line.strip(',').split(',')
            edit_list[items[1].lower()] = list(map(lambda x: x.lower(), items[2:]))
    all_t = set()

    def make_topics(topics):
        result = []
        for t in topics.split(','):
            if t.strip() == '':
                continue
            if t in edit_list:
                all_t.add(edit_list[t][0])
                if edit_list[t][0] == '-1':
                    continue
                elif edit_list[t][0] == '-2':
                    result.append(t)
                    continue
                result.extend(edit_list[t])
            else:
                result.append(t)
        return ",".join(list(set(result)))

    df.topics = df.topics.apply(make_topics)
    df = df[df.topics != '']
    return df

def aggregate_topics(file_name, df):
    """The N:1 co-occurrence merge - `abbr.csv` only.

    A rule fires only when the repo carries *every* topic on the right-hand side, and then
    replaces all of them with the single topic in column 1.

        'abbr.csv'   line: `1,zsh,oh-my-zsh,zsh,`
                     -> edit_list entry (['oh-my-zsh', 'zsh'], 'zsh')

        topics  'oh-my-zsh,zsh,terminal'  -> 'zsh,terminal'      both matched, merged
        topics  'oh-my-zsh,terminal'      -> 'oh-my-zsh,terminal'  only one present, no-op

    The running example carries no such pair, so this pass leaves it untouched.
    """
    edit_list = []
    with open(file_name) as file:
        for line in file.read().strip('\n').split('\n'):
            items = line.strip(',').split(',')
            # edit_list.append(([ps.stem(x) for x in items[2:]],ps.stem(items[1])))
            edit_list.append(([x.lower().strip() for x in items[2:]], items[1].lower().strip()))
    all_t = set()

    def make_topics(topics):
        result = []
        deleted = []
        # topics = [ps.stem(x) for x in topics.split(',')]
        topics = [x for x in topics.split(',')]
        for abr in edit_list:
            if set(abr[0]) & set(topics) == set(abr[0]):
                all_t.add(abr[1])
                deleted.extend(list(abr[0]))
                result.append(abr[1])
        for del_topics in set(deleted):
            topics.remove(del_topics)
        result.extend(topics)
        return ",".join(list(set(result)))

    df.topics = df.topics.apply(make_topics)

    return df

def aggregate_two_topics(file_name, df):
    """The 1:1 alias map - `replace.csv` only.

    Careful with the direction: the key is column *2* and the value is column *1*, so the
    rule reads right-to-left, the opposite of `apply_csv`.

        'replace.csv'  line: `1,vue,vuex`     -> edit_list['vuex'] = 'vue'
                       line: `7,react,reactstrap` -> edit_list['reactstrap'] = 'react'
                       line: `424,learn,learning`  -> edit_list['learning'] = 'learn'

        topics  'vuex,javascript'  ->  'vue,javascript'

    The running example already holds 'learn' (the target, not the key) and 'react', so this
    pass is a no-op for it.
    """
    edit_list = {}
    with open(file_name) as file:
        for line in file.read().strip('\n').split('\n'):
            items = line.rstrip(',').split(',')
            # edit_list.append(([ps.stem(x) for x in items[2:]],ps.stem(items[1])))
            edit_list[items[2].lower().strip()] = items[1].lower().strip()
    all_t = set()

    def make_topics(topics):
        result = []
        topics = [x for x in topics.split(',')]
        for topic in topics:
            if topic in edit_list:
                all_t.add(topic)
                result.append(edit_list[topic])
            else:
                result.append(topic)
        return ",".join(list(set(result)))

    df.topics = df.topics.apply(make_topics)
    return df


def preprocess(df, MIN_TOPIC_LENGTH=4):
    """Run the 13 rewriting passes, bracketed by a star filter and two list-shaping steps.

    Traced on the running example, `freeCodeCamp/freeCodeCamp`, showing only the passes that
    actually change it (10 topics in, 7 out):

        raw       learn-to-code nonprofits programming nodejs react d3 careers education teachers javascript
        split     learn to code nonprofits programming nodejs react d3 careers education teachers javascript
        plural    learn to code nonprofit  programming nodejs react d3 career  education teacher  javascript
        low_freq  learn to code            programming nodejs react d3         education teacher  javascript
        stopword  learn    code            programming nodejs react d3         education teacher  javascript
        delete    learn                                nodejs react d3         education teacher  javascript

        result    ['learn', 'nodejs', 'react', 'd3', 'education', 'teacher', 'javascript']

    Note how order matters between passes: `careers` is singularized to `career` by
    remove_plural_topics, and only then does low_freq_topics_1's `40321,career,-1` rule reach
    it. Run in the other order, `careers` would survive.
    """
    # Keeps the pre-normalization topics for comparison; nothing downstream reads them.
    #     original_topics  'learn-to-code,nonprofits,programming,...'
    df['original_topics'] = df.topics

    # Intended as a popularity floor, but the column is misaligned: main.py reads the CSV as
    # repo,fork,open_issue,star while the file is actually repo,stars,forks,issues, so
    # df['stars'] holds the ISSUE count. For freeCodeCamp that is 18688, not 408888.
    # Filtering on issues still admits every popular repo, so the effect is mild - see
    # docs/ARCHITECTURE.md section 4, item 13.
    df = df[df['stars'] >= 10]

    # KG_NORMALISE_STAGE=pre runs the canonicalisation BEFORE the hand-built rules, where the
    # graph still has raw surface variants to act on. Running it after (the default) leaves it
    # almost nothing to do, because the 765 alias rules below have already merged them.
    if os.environ.get("KG_NORMALISE") == "1" and os.environ.get("KG_NORMALISE_STAGE") == "pre":
        df = kg_normalise_topics_string(df)

    # 1. strip version suffixes           `23,netflow-v5,netflow`     'netflow-v5' -> 'netflow'
    df = apply_csv(f'{REWRITING_RULE_FOLDER}/topics_contains_version.csv', df)
    # 2. strip embedded digits             `9386,python3,python`       'python3' -> 'python'
    #    Some rules map a topic to itself:  `855,d3,d3`                 'd3' -> 'd3' (no-op)
    df = apply_csv(f'{REWRITING_RULE_FOLDER}/topics_contains_number.csv', df)
    # 3. split hyphenated topics           `204,learn-to-code,learn,to,code`
    #    example:  'learn-to-code' -> 'learn' + 'to' + 'code'   (10 topics -> 12)
    df = apply_csv(f'{REWRITING_RULE_FOLDER}/split_dash_topics.csv',df)
    # 4. split run-together topics         `1,rbot,bot`                'rbot' -> 'bot'
    df = apply_csv(f'{REWRITING_RULE_FOLDER}/contains_top_topics.csv',df)
    # 5. singularize                       `25,careers,career`         'careers' -> 'career'
    #    example:  'careers' -> 'career', 'teachers' -> 'teacher', 'nonprofits' -> 'nonprofit'
    df = apply_csv(f'{REWRITING_RULE_FOLDER}/remove_plural_topics.csv', df)
    # 6. drop rare topics                  `40321,career,-1`           'career' -> (gone)
    #    example:  'career' and 'nonprofit' are dropped here  (12 -> 10)
    df = apply_csv(f'{REWRITING_RULE_FOLDER}/low_freq_topics_1.csv', df)
    # 7. drop stopword topics              `2390,to,-1`                'to' -> (gone)
    #    example:  the 'to' produced by pass 3 is removed here  (10 -> 9)
    df = apply_csv(f'{REWRITING_RULE_FOLDER}/remove_stopwords_topic.csv',df)
    # 8. lemmatize                         `21,watches,watch`          'watches' -> 'watch'
    df = apply_csv(f'{REWRITING_RULE_FOLDER}/remove_lemmatize_topic.csv',df)
    # 9. blacklist                         `36,programming,-1`         'programming' -> (gone)
    #    example:  'programming' and 'code' are dropped here  (9 -> 7)
    df = apply_csv(f'{REWRITING_RULE_FOLDER}/delete.csv',df)
    # 10. 1:1 alias, read right-to-left    `1,vue,vuex`                'vuex' -> 'vue'
    df = aggregate_two_topics(f'{REWRITING_RULE_FOLDER}/replace.csv', df)
    # 11. N:1 co-occurrence merge          `1,zsh,oh-my-zsh,zsh`       {'oh-my-zsh','zsh'} -> 'zsh'
    df = aggregate_topics(f'{REWRITING_RULE_FOLDER}/abbr.csv', df)
    # 12. expand abbreviations             `1,js,javascript`           'js' -> 'javascript'
    df = apply_csv(f'{REWRITING_RULE_FOLDER}/contractions.csv',df)
    # 13. curated splits                   `280,homeassistant,assistant,home`
    df = apply_csv(f'{REWRITING_RULE_FOLDER}/contains_selected_topics.csv',df)

    # Keep only repos with MORE than 4 topics - a repo with exactly 4 is dropped. This is why
    # nothing downstream ever sees a cold-start repository, despite the seeding design.
    #     'learn,nodejs,react,d3,education,teacher,javascript'  -> 7 topics, kept
    #     'python,api'                                          -> 2 topics, dropped
    clear_dataset = df[df.apply(lambda x: len(x['topics'].split(",")) > MIN_TOPIC_LENGTH, axis=1)]

    # str -> list. Everything after this point expects `topics` to be a list, not a string.
    normalised = topics_string_to_list(clear_dataset)
    if os.environ.get("KG_NORMALISE_STAGE", "post") == "post":
        normalised = kg_normalise_topics(normalised)
    return normalised


def kg_normalise_topics_string(df):
    """Canonicalise while `topics` is still a comma-joined string (pre-rules stage)."""
    from kg_normalise import load_vocabulary, normalise_topic_list
    vocabulary = load_vocabulary()
    before = {t for ts in df.topics for t in ts.split(',') if t.strip()}

    def norm(topics_str):
        topics = [t.strip() for t in topics_str.split(',') if t.strip()]
        return ",".join(normalise_topic_list(topics, vocabulary))

    df.topics = df.topics.apply(norm)
    after = {t for ts in df.topics for t in ts.split(',') if t.strip()}
    print(f"KG normalisation (pre-rules): {len(before)} -> {len(after)} distinct topics")
    return df

def kg_normalise_topics(df):
    """Optional final pass: collapse topics onto the SED-KGraph vocabulary.

    Off unless KG_NORMALISE=1. Runs last so it sees the output of the hand-built alias
    rules rather than competing with them.
    """
    if os.environ.get("KG_NORMALISE") != "1":
        return df
    from kg_normalise import load_vocabulary, normalise_topic_list
    vocabulary = load_vocabulary()
    before = {t for ts in df.topics for t in ts}
    df.topics = df.topics.apply(lambda ts: normalise_topic_list(ts, vocabulary))
    after = {t for ts in df.topics for t in ts}
    print(f"KG normalisation: {len(before)} -> {len(after)} distinct topics")
    return df
