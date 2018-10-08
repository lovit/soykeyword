# Python library for Keyword Extraction

키워드 / 연관어 추출을 위한 파이썬 라이브러리 입니다. by [Lovit (Hyunjoong)][lovit] and [Hunsik Shin][hunsik] 

soykeyword 에서 추출하는 키워드와 연관어는 다음과 같이 정의됩니다. 한 문서 집합의 **키워드**는 다른 문서 집합과 해당 문서 집합을 구분할 수 있는 질 좋은 단어이며 (구분력, discriminative power), 해당 집합을 잘 설명할 수 있는 (설명력, high coverage) 단어입니다. 빈도수가 낮은 단어는 한 집합에서만 등장할 가능성이 높기 때문에 구분력은 크지만 설명력이 약합니다. 제안된 두 가지 알고리즘은 높은 설명력과 구분력을 동시에 지니는 단어들을 키워드로 선택합니다. 

**연관어**는 기준 단어가 포함된 문서 집합과 포함되지 않은 문서 집합을 구분하는 키워드를 연관어로 정의합니다. 이는 co-occurrence 가 높은 단어라는 의미이기도 합니다. co-occurrence 가 높으면서도 설명력이 좋은 단어를 선택합니다. 



## Setup

- pip install soykeyword

## Requires

- Python >= 3.4 (not tested in Python 2)
- numpy >= 1.12.1
- scikit-learn >= 0.18
- psutil >=5.0.1

## Usage

### Lasso Regerssion Keyword Extractor

학습은 sparse matrix x 를 extractor 에 입력합니다. index2word 는 word idx 에 대한 단어 list 형식입니다. 이를 train() 에 입력하지 않으면 키워드와 연관어가 단어가 아닌 word idx 로 출력됩니다.

    from soykeyword.lasso import LassoKeywordExtractor

    lassobased_extractor = LassoKeywordExtractor(min_tf=20, min_df=10)
    lassobased_extractor.train(x, index2word) # x: sparse matrix

키워드를 추출할 문서 집합 documents 를 extract_from_docs() 에 입력하면, 해당 문서 집합과 그 외의 문서 집합을 구분하는 keywords 를 추출합니다. 

    keywords = lassobased_extractor.extract_from_docs(
        documents, 
        min_num_of_keywords=30
    )

연관어는 extract_from_word 에 단어를 입력하면 됩니다.

    lassobased_extractor.extract_from_word(
        '아이오아이',
        min_num_of_keywords=30
    )

하루 뉴스를 기준으로 '아이오아이'의 연관어를 추출한 예시입니다.

    [KeywordScore(word='아이오아이', frequency=270, coefficient=17.850189941320671),
     KeywordScore(word='엠카운트다운', frequency=221, coefficient=1.200759338786378),
     KeywordScore(word='뮤직', frequency=195, coefficient=1.081777863860977),
     KeywordScore(word='일산동구', frequency=36, coefficient=0.98636875892070186),
     KeywordScore(word='키미', frequency=297, coefficient=0.70877507721215616),
     KeywordScore(word='챔피언', frequency=105, coefficient=0.51940928356916138),
     KeywordScore(word='강렬', frequency=352, coefficient=0.36972563098092176),
     KeywordScore(word='컴백', frequency=536, coefficient=0.30677481146665397),
     KeywordScore(word='화려', frequency=518, coefficient=0.26764304959838653),
     KeywordScore(word='수출', frequency=735, coefficient=0.23882691530127598),
     KeywordScore(word='걸그룹', frequency=1060, coefficient=0.20972098801573957),
     KeywordScore(word='방영', frequency=208, coefficient=0.19694219657704334),
     KeywordScore(word='프로듀스101', frequency=96, coefficient=0.17074232136595247),
     ...

자세한 튜토리얼은 [링크][lasso_tutorial]에 있습니다.

### Proportion based Keyword Extractor

Proportion based 키워드 / 연관어 추출은 두 집합의 단어 출연 확률의 비율을 바탕으로 키워드를 추출합니다. P(w|pos) 는 키워드를 추출할 문서 집합에서의 단어 w 의 출연 비율이며, P(w|neg)는 그 외의 문서 집합에서의 단어 w의 출연 비율 입니다. 

score(w) = P(w|pos) / { P(w|pos) + P(w|neg) }

학습 데이터의 형태는 (sparse matrix, index2word) 혹은 텍스트 데이터, 두 종류를 모두 지원합니다. 

텍스트 데이터 형식으로 학습을 할 경우에는 min_tf, min_df, tokenize 를 설정해줍니다. 다음의 예시는 default value 입니다.

    from soykeyword.proportion import CorpusbasedKeywordExtractor
    corpusbased_extractor = CorpusbasedKeywordExtractor(
        min_tf=20,
        min_df=2,
        tokenize=lambda x:x.strip().split(),
        verbose=True
    )

    # docs: list of str like
    corpusbased_extractor.train(docs)

키워드를 추출할 문서 집합 documents 를 입력합니다.

    keywords = corpusbased_extractor.extract_from_docs(
        documents,
        min_score=0.8,
        min_frequency=100
    )

연관어를 추출할 단어 word 를 입력합니다. 

    keywords = corpusbased_extractor.extract_from_word(
        '아이오아이',
        min_score=0.8,
        min_frequency=100
    )

하루의 뉴스를 바탕으로 추출한 아이오아이의 연관어 입니다. 

    keywords[:10]

    [KeywordScore(word='아이오아이', frequency=270, score=1.0),
     KeywordScore(word='엠카운트다운', frequency=221, score=0.997897148491129),
     KeywordScore(word='펜타곤', frequency=104, score=0.9936420169665052),
     KeywordScore(word='잠깐', frequency=162, score=0.9931809154109712),
     KeywordScore(word='엠넷', frequency=125, score=0.9910325251765126),
     KeywordScore(word='걸크러쉬', frequency=111, score=0.9904705029926091),
     KeywordScore(word='타이틀곡', frequency=311, score=0.987384461584851),
     KeywordScore(word='코드', frequency=105, score=0.9871835929954923),
     KeywordScore(word='본명', frequency=105, score=0.9863934667369743),
     KeywordScore(word='엑스', frequency=101, score=0.9852544036088814)]

학습데이터의 형태가 (sparse matrix, index2word) 라면 MatrixbasedKeywordExtractor 를 이용합니다.

    from soykeyword.proportion import MatrixbasedKeywordExtractor

    matrixbased_extractor = MatrixbasedKeywordExtractor(
        min_tf=20,
        min_df=2,
        verbose=True
    )

    matrixbased_extractor.train(x, index2word)

자세한 튜토리얼은 [링크][proportion_tutorial]에 있습니다.

## 함께 이용하면 좋은 라이브러리들

### soynlp

한국어 자연어처리를 위한 미등록단어 문제 해결을 위한 단어 추출 / 단어 추출기의 학습 결과를 이용하는 토크나이저 / 품사 판별 / 정규화 를 지원합니다.

- https://github.com/lovit/soynlp
- pip install soynlp

### KoNLPy

KoNLPy 는 한국어 정보처리를 위한 파이썬 패키지입니다. 한나눔, 꼬꼬마, 코모란, MeCab-ko, 트위터 한국어 분석기를 파이썬 환경에서 제공합니다. 

- http://konlpy.org
- KoNLPy 는 Java를 이용하기 때문에 Java 와 JPype 가 필요합니다. 홈페이지의 설치법을 반드시 보시기 바랍니다. 

### customized KoNLPy

KoNLPy 에 등록되지 않은 단어를 손쉽게 처리하기 위하여 템플릿과 사전 기반 string match 를 KoNLPy 와 함께 이용하는 wrapping 파이썬 패키지입니다.

- https://github.com/lovit/customized_konlpy
- pip install customized_konlpy

### soyspacing

띄어쓰기 오류가 있을 경우 이를 제거하면 텍스트 분석이 쉬워질 수 있습니다. 분석하려는 데이터를 기반으로 띄어쓰기 엔진을 학습하고, 이를 이용하여 띄어쓰기 오류를 교정합니다. 

- https://github.com/lovit/soyspacing
- pip install soyspacing

[lovit]: https://github.com/lovit
[hunsik]: https://github.com/hunsik
[lasso_tutorial]: tutorials/keyword_extraction_using_lasso_regression.ipynb
[proportion_tutorial]: tutorials/keyword_extraction_using_proportion_ratio.ipynb
