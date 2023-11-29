""" Copyright Xplainable Pty Ltd, 2023"""

from ...utils.collections import stopwords
from ...utils.dualdict import TargetMap
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import re
import pandas as pd
import numpy as np


class NLPExtractor:
    """Converts a single NLP feature into multiple quantitative features.

    Args:
        urls (bool, optional): Extract URLs from text. Defaults to True.
        chars (bool, optional): Count characters in text. Defaults to True.
        entities (bool, optional): Extract named entities. Defaults to False.
        words (bool, optional): Count words in text. Defaults to True.
        sentiment (bool, optional): Estimate text sentiment. Defaults to True.
    """

    def __init__(self, urls=True, uppercase=True, punctuation=True, chars=True,\
        numbers=True, emojis=True, words=True, sentiment=True, \
            drop_stopwords=True, ngrams=1):

        super().__init__()

        # Instantiate params
        self.urls = urls
        self.uppercase = uppercase
        self.punctuation = punctuation
        self.chars = chars
        self.numbers = numbers
        self.emojis = emojis
        self.words = words
        self.sentiment = sentiment
        self.drop_stopwords = drop_stopwords
        self.ngrams = ngrams

        # Load spacy model if required
        # if entities:
        #     self.nlp = spacy.load('en_core_web_sm', disable=[])

        # Load stop words for text parsing
        self.stop_words = set(stopwords)

        # Instantiate meta data stores
        self.base_value = None
        self.__meta = pd.DataFrame()
        self.__ngram_meta = pd.DataFrame()
        self.word_map = {}
        self.ngram_map = {}
        self.target_map = {}

    def _count_uppercase_words(self, s):
        return len(re.findall(r'\b[A-Z]+\b', s))

    def _count_punctuation(self, s):
        return len(re.findall(r'[A-Z]', s))

    def _count_numbers(self, s):
        return len(re.findall(r'\d', s))

    def _count_emojis(self, s):

        emoji_pattern = re.compile("["
        u"\U0001f600-\U0001f64f"  # emoticons
        u"\U0001f300-\U0001f5ff"  # symbols & pictographs
        u"\U0001f680-\U0001f6ff"  # transport & map symbols
        u"\U0001f1e0-\U0001f1ff"  # flags (iOS)
                           "]+", flags=re.UNICODE)

        return len(emoji_pattern.findall(s))

    def _clean_string(self, s):
        """ Strips non-text characters from a string.

        Args:
            s (str): The string to be cleaned.

        Returns:
            str: The cleaned string.
        """

        text = re.sub("[^a-zA-Z']", ' ', str(s))
        text = re.sub("[^a-zA-Z ]", '', str(text))

        return text

    def _split_column(self, ser):
        """ Splits a text column into individual lowercase words.

        Args:
            ser (pd.Series): The text column to be split.

        Returns:
            pd.Series: The transformed column.
        """

        ser = ser.str.lower().str.split()

        return ser

    def _lemmatize(self, text):
        """ Lemmatizes a list of strings and removes stopwords.

        Args:
            text (list): The list of words to be lemmatized.

        Returns:
            list : The list of lemmatized words.
        """

        # Instantiate lemmatizer
        wnl = WordNetLemmatizer()

        # Apply lemmatization
        stems = [wnl.lemmatize(i) for i in text]

        return stems

    def _remove_stopwords(self, text):

        return [i for i in text if i not in self.stop_words]

    def _chunker(self, iterable, total_length, chunk_size):
        """ Converts an interable into chunks for parrallelism.

        Args:
            iterable (iter): An iterable object to be chunked.
            total_length (int): The total length of the iterable.
            chunk_size (int): The size of the chunks.

        Returns:
            generator: A generator containing each chunk.
        """

        return (iterable[pos: pos + chunk_size] for pos in
                range(0, total_length, chunk_size))

    def _flatten(self, lists):
        """Flatten a list of lists to a combined list.

        Args:
            lists (list): A list containing lists.

        Returns:
            list: The flattened list.
        """

        return [item for sublist in lists for item in sublist]

    # def _get_named_entities(self, text):
    #     """ Extracts named entities from a string.

    #     Args:
    #         text (str): A non-empty string to be processed.

    #     Returns:
    #         collections.Counter: The counts of each named entity type.
    #     """

    #     # Get all named entities from string
    #     ne = [ent.label_ for ent in self.nlp(text).ents]

    #     return Counter(ne)

    # def batch_process(self, texts):
    #     """ Batch processes named entities recognition.

    #     Args:
    #         texts (pd.Series): A series containing strings to be processed.

    #     Returns:
    #         list: A list of processed batches.
    #     """

    #     # Instantiate pipeline
    #     _pipe = []

    #     # Add batches to pipeline
    #     for doc in self.nlp.pipe(texts, batch_size=100):
    #         _pipe.append(self._get_named_entities(str(doc)))

    #     return _pipe

    # def _preprocess_parallel(self, texts, chunk_size=200):
    #     """ Distributes NLP pipeline processes across all available cores.

    #     Args:
    #         texts (pd.Series): A series of texts to be processed.
    #         chunk_size (int, optional): Size of chunks. Defaults to 200.

    #     Returns:
    #         list: Flattened list of results.
    #     """

    #     # Get the number of machine cores
    #     cpu_count = psutil.cpu_count(logical=False)

    #     print(f'Running on {cpu_count} CPUs.')

    #     # Instantiate the executor
    #     executor = Parallel(n_jobs=cpu_count,
    #                         backend='multiprocessing',
    #                         prefer="processes")

    #     # Prepare batch processing
    #     do = delayed(self.batch_process)

    #     # Process chunks in batches
    #     tasks = (do(chunk) for chunk in
    #              self._chunker(texts, len(texts), chunk_size=chunk_size))
    #     result = executor(tasks)

    #     return self._flatten(result)

    def _count_upper(self, text):
        """ Counts uppercase letters in a string.

        Args:
            text (str): The string to be searched.

        Returns:
            int: The number of uppercase letters in string.
        """

        return len(re.findall(r'[A-Z]', text))

    def _url_count(self, text):
        """Counts URLs within a string.

        Args:
            text (string): The string to be searched.

        Returns:
            int: The number of URLs contained in the string.
        """

        return len(re.findall(r"(?P<url>https?://[^\s]+)", str(text)))

    def _url_drop(self, text):
        """ Drops URLs from a string.

        Args:
            text (str): The string to be processed.

        Returns:
            str: The string with URLs removed.
        """
        # Find and replace URLs with empty string
        for string in re.findall(r"(?P<url>https?://[^\s]+)", str(text)):
            text = text.replace(string, "")

        return text

    def _generate_ngrams(self, words, n):
        """ Constructs a list of n-grams from a string.

        Args:
            text (str): The string to be processed.
            n (int): The number of n-grams.

        Returns:
            list: List of n-grams.
        """

        # Instantiate output
        ngrams = []

        # Iterate through words and store n-grams
        for i in range(len(words) - n + 1):
            ngrams.append(" ".join(words[i:i + n]).lower())

        return ngrams

    def _get_word_map(self, x, y, ngram_range=(1,1), stop_words=None):
        """ Maps words with their relationship with a target feature.

        Args:
            x (pandas.Series): The text feature to be analysed.
            y (pandas.Series): The target feature.
            min_word_freq (float, optional): Min required freq. Defaults to 0.0003.

        Returns:
            dict: A dictionary map of words to scores.
        """

        # Instantiate the CountVectorizer
        vectorizer = CountVectorizer(
            analyzer='word',
            ngram_range=ngram_range,
            stop_words=stop_words)

        # Fit the text column to the CountVectorizer
        x_vals = vectorizer.fit_transform(
            [" ".join(inner_list) for inner_list in x.to_list()])

        # Get the list of words
        feature_list = vectorizer.get_feature_names_out()

        # Convert the values to a dataframe
        df = pd.DataFrame(x_vals.toarray(), columns=feature_list)

        # Assign the target to the new dataframe
        df['target'] = y.values

        # Calculate the pct of positive class for each feature/word
        feature_vals = [df.loc[(df[col] > 0), 'target'].mean() for col in
                        df.drop(columns=['target']).columns]

        # Convert 0's to np.nans
        df = df.replace(0, np.nan)

        # Calculate the frequency of each word across observations
        f = df.drop(columns=['target']).count(axis=0) / len(df)
        frequency = pd.DataFrame(f)

        # Assign the feature values to the word map
        frequency['word_map'] = feature_vals
        frequency.columns = ['freq', 'word_map']

        # Calculate difference to base_value
        frequency['word_map'] = frequency['word_map'] - self.base_value

        # Filter dataframe to show only above threshold frequency
        __meta = frequency[['word_map', 'freq']]

        # Convert scores to dictionary format
    
        return __meta

    def _score_sentiment(self, words):
        """ Scores the sentiment of a sentence from the fitted model.

        Args:
            words (list): A list of words/ngrams.

        Returns:
            int: The sentiment score.
        """

        text = ' '.join(words)
        score = 0

        # Instantiate seen set
        n = self.ngrams
        while n > 1:
            ngrams = self._generate_ngrams(words, n)

            for gram in ngrams:
                if gram in self.word_map.keys():
                    s = self.word_map[gram]
                    score += s
                    text = text.replace(gram, '')

            n -= 1

        # Assign score to each individual word/ngram and sum  
        stripped_words = [i.strip() for i in text.split()]
        s = np.nansum([self.word_map[i] for i in stripped_words if \
            i in self.word_map])

        score += s
        
        return score

    def _score_pos_sentiment(self, words):
        """ Scores the sentiment of a sentence from the fitted model.

        Args:
            words (list): A list of words/ngrams.

        Returns:
            int: The sentiment score.
        """

        text = ' '.join(words)
        score = 0

        # Instantiate seen set
        n = self.ngrams
        while n > 1:
            ngrams = self._generate_ngrams(words, n)

            for gram in ngrams:
                if gram in self.word_map.keys():
                    s = self.word_map[gram]
                    if s > 0:
                        score += s
                        text = text.replace(gram, '')

            n -= 1

        # Assign score to each individual word/ngram and sum  
        stripped_words = [i.strip() for i in text.split()]
        s = np.nansum([self.word_map[i] for i in stripped_words if \
            i in self.word_map and self.word_map[i] > 0])

        score += s
        
        return score

    def _count_pos_sentiment(self, words):
        """ Scores the sentiment of a sentence from the fitted model.

        Args:
            words (list): A list of words/ngrams.

        Returns:
            int: The sentiment score.
        """

        text = ' '.join(words)
        score = 0

        # Instantiate seen set
        n = self.ngrams
        while n > 1:
            ngrams = self._generate_ngrams(words, n)

            for gram in ngrams:
                if gram in self.word_map.keys():
                    s = self.word_map[gram]
                    if s > 0:
                        score += 1
                        text = text.replace(gram, '')

            n -= 1

        # Assign score to each individual word/ngram and sum  
        stripped_words = [i.strip() for i in text.split()]
        s = np.nansum([1 for i in stripped_words if \
            i in self.word_map and self.word_map[i] > 0])

        score += s
        
        return score

    def _score_neg_sentiment(self, words):
        """ Scores the sentiment of a sentence from the fitted model.

        Args:
            words (list): A list of words/ngrams.

        Returns:
            int: The sentiment score.
        """

        text = ' '.join(words)
        score = 0

        # Instantiate seen set
        n = self.ngrams
        while n > 1:
            ngrams = self._generate_ngrams(words, n)

            for gram in ngrams:
                if gram in self.word_map.keys():
                    s = self.word_map[gram]
                    if s < 0:
                        score += s
                        text = text.replace(gram, '')

            n -= 1

        # Assign score to each individual word/ngram and sum  
        stripped_words = [i.strip() for i in text.split()]
        s = np.nansum([self.word_map[i] for i in stripped_words if \
            i in self.word_map and self.word_map[i] < 0])

        score += s
        
        return score

    def _count_neg_sentiment(self, words):
        """ Scores the sentiment of a sentence from the fitted model.

        Args:
            words (list): A list of words/ngrams.

        Returns:
            int: The sentiment score.
        """

        # Instantiate seen set
        text = ' '.join(words)
        score = 0

        # Instantiate seen set
        n = self.ngrams
        while n > 1:
            ngrams = self._generate_ngrams(words, n)

            for gram in ngrams:
                if gram in self.word_map.keys():
                    s = self.word_map[gram]
                    if s > 0:
                        score += 1
                        text = text.replace(gram, '')

            n -= 1

        # Assign score to each individual word/ngram and sum  
        stripped_words = [i.strip() for i in text.split()]
        s = np.nansum([1 for i in stripped_words if \
            i in self.word_map and self.word_map[i] < 0])

        score += s
        
        return score

    def fit(self, x, y):
        """ Extracts and stores nlp data with respect to y.

        Args:
            x (pd.Series): The text feature to be processed.
            y (pd.Series): The target feature.
            min_word_freq (float, optional): Min required freq. Defaults to 0.0003.

        Returns:
            self
        """

        # Encode target categories if not numeric
        if y.dtype == 'object':

            # Cast as category
            target_ = y.astype('category')

            # Get the inverse label map
            self.target_map = TargetMap(dict(enumerate(target_.cat.categories)), True)

            # Encode the labels
            y = y.map(self.target_map)

        # Calculate the feature base value
        self.base_value = y.mean()

        # Only need to fit word map if calculating sentiment
        if self.sentiment:

            # Copy data and reset index
            x = x.copy(deep=True)

            # Clean strings
            x = x.apply(self._url_drop)
            x = x.apply(self._clean_string)

            # Split and lemmatize words
            x = self._split_column(x)#.apply(self._lemmatize) ignoring
            if self.drop_stopwords:
                x = x.apply(self._remove_stopwords)

            # Build word map
            if self.ngrams > 1:
                print("Generating ngram profile")
                self.__ngram_meta = self._get_word_map(
                    x, y,
                    ngram_range=(2, self.ngrams)
                    )

                x = x.apply(self._remove_ngrams)

            print("Generating word profile")
            self.__meta = self._get_word_map(
                x, y,
                stop_words='english')

            self.set_word_map()

        return self

    def _remove_ngrams(self, x):
        text = ' '.join(x)
        for k in self.ngram_map.keys():
            text = text.replace(k, '')
        text = text.strip()
        
        return text.split(' ')

    def _filter_word_map(self, meta, min_freq, max_freq):
        df = meta[(max_freq > meta['freq']) & (meta['freq'] > min_freq)]
        return df['word_map'].to_dict()

    def set_word_map(
            self, min_word_freq=0.002, max_word_freq=0.7, min_ngram_freq=0.002,
            max_ngram_freq=0.3):
        
        word_map = {}
        if len(self.__ngram_meta) > 0:
            word_map.update(
                self._filter_word_map(
                    self.__ngram_meta, min_ngram_freq, max_ngram_freq
                    ))
        if len(self.__meta) > 0:
            word_map.update(
                self._filter_word_map(
                    self.__meta, min_word_freq, max_word_freq
                    ))

        self.word_map = word_map
        
        return
            
    def transform(self, x):
        """ Extracts NLP features from text column.

        Args:
            x (pd.Series): The text feature to be processed.
            parallelise (bool, optional): Distributes NER. Defaults to False.
            
        Returns:
            pd.DataFrame: A dataframe containing the NLP features.
        """

        # Copy x
        x = x.copy(deep=True)

        # Instantiate output df
        df = pd.DataFrame(x)

        # Set word map limits
        #self.set_word_map(
        # min_word_freq, max_word_freq, min_ngram_freq, max_ngram_freq)

        # Count and remove urls
        if self.urls:
            df['nlp_urls'] = x.apply(self._url_count)
            x = x.apply(self._url_drop)

        if self.uppercase:
            df['nlp_uppercase_words'] = x.apply(self._count_uppercase_words)

        if self.punctuation:
            df['nlp_punctuations'] = x.apply(self._count_punctuation)

        if self.emojis:
            df['nlp_emojis'] = x.apply(self._count_emojis)
            
        if self.numbers:
            df['nlp_numbers'] = x.apply(self._count_numbers)

        # Clean the text
        x = x.apply(self._clean_string)

        # Count the number of characters
        if self.chars:
            df['nlp_char'] = x.apply(len)

        # Extract named entities with spacy model
        # if self.entities:

        #     # Normal extraction
        #     if not parallelise:
        #         print('Running on single CPU.')
        #         df['nlp_entities'] = x.apply(
        #             self._get_named_entities)

        #     # Distributed extraction for large datasets
        #     else:
        #         df['nlp_entities'] = \
        #             self._preprocess_parallel(x, chunk_size=1024)

        # Split and lemmatize words
        x = self._split_column(x)#.apply(self._lemmatize)
        if self.drop_stopwords:
            x = x.apply(self._remove_stopwords)

        # Count words
        if self.words:
            df['nlp_words'] = x.apply(len)

        if self.chars and self.words:
            df['nlp_avg_word_len'] = df['nlp_char'] / df['nlp_words']
            df['nlp_avg_word_len'] = df['nlp_avg_word_len'].replace(np.inf, 0)

        # Convert named entities to individual features
        # if self.entities:

        #     # Record the initial column names
        #     init_cols = df.columns

        #     # Create feature for each named entity
        #     df = pd.concat([df.drop('nlp_entities', axis=1), pd.DataFrame(
        #         df['nlp_entities'].tolist())], axis=1)

        #     # Get the list of entity names
        #     ent_list = [col for col in df if col not in init_cols]

        #     # Drop the feature if less than 10% of rows contain the entity
        #     drop_col = [col for col, val in df[
        #         ent_list].sum().iteritems() if val < 0.1 * len(df)]

        #     df = df.drop(drop_col, axis=1)

        #     # Fill remaining entities' na with 0
        #     keep_col = [col for col in ent_list if col not in drop_col]
        #     df[keep_col] = df[keep_col].fillna(0)

        #     # rename features to prefix with 'nlp_'
        #     rename_cols = {ent: f'nlp_{ent.lower()}' for ent in ent_list}
        #     df = df.rename(rename_cols)

        # Calculate text sentiment
        if self.sentiment:
            #df['nlp_sentiment'] = x.apply(self._score_sentiment)
            df['nlp_pos'] = x.apply(self._score_pos_sentiment)
            df['nlp_pos_count'] = x.apply(self._count_pos_sentiment)
            df['nlp_neg'] = x.apply(self._score_neg_sentiment)
            df['nlp_neg_count'] = x.apply(self._count_neg_sentiment)
            

        # Drop original text column
        df.drop(columns=[x.name], inplace=True)

        return df

    def map_scores(self, text):
        """ Maps word scores to a sentence.

        Args:
            text (str): The sentence to score

        Returns:
            dict: Word mappings
        """

        # Apply test cleaning and lemmatization
        
        text = self._url_drop(text)
        text = self._clean_string(text)
        tokens = text.lower().split()
        if self.ngrams > 1:
            ngrams = self._generate_ngrams(text.split(), self.ngrams)
            tokens = ngrams + self._lemmatize(tokens)

        # Instantiate the output map
        outmap = []

        seen = set()

        # Apply word mappings
        for ngram in tokens:
            if ' ' in ngram:
                seen = seen.union(set(ngram.split()))
            
            if ngram in self.word_map:
                if ngram in seen:
                    outmap.append((ngram,  0))
                    seen.remove(ngram)
                    
                else:
                    outmap.append((ngram, self.word_map[ngram]))
            else:
                outmap.append((ngram,  0))

        return outmap
