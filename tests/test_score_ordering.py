from hashformers.beamsearch.minicons_lm import MiniconsLM


class Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class TokenScores:
    def __init__(self, total):
        self.total = total

    def sum(self, _dimension):
        return Scalar(self.total)


class MaskedScorer:
    def sequence_score(self, _batch, reduction):
        return [reduction(TokenScores(-5)), reduction(TokenScores(-20))]


class Seq2SeqScorer:
    def sequence_score(self, _batch, source_format):
        assert source_format == "blank"
        return [-5, -20]


def build_lm(model_type, scorer):
    lm = object.__new__(MiniconsLM)
    lm.model_type = model_type
    lm.scorer = scorer
    return lm


def test_masked_scores_are_lower_for_more_probable_candidates():
    lm = build_lm("MaskedLMScorer", MaskedScorer())

    assert lm.get_batch_scores(["more probable", "less probable"]) == [5, 20]


def test_seq2seq_scores_are_lower_for_more_probable_candidates():
    lm = build_lm("Seq2SeqScorer", Seq2SeqScorer())

    assert lm.get_batch_scores(["more probable", "less probable"]) == [5, 20]
