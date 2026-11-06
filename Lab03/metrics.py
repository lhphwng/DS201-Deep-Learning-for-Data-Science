from collections import defaultdict

from sklearn.metrics import f1_score


# ---------------------------------------------------------------------------
# Phân loại văn bản
# ---------------------------------------------------------------------------
def f1_classification(gts: list, pds: list, average: str = "macro") -> float:
    return float(f1_score(gts, pds, average=average, zero_division=0))


# ---------------------------------------------------------------------------
# Nhận diện thực thể (BIO)
# ---------------------------------------------------------------------------
def get_entities(tags: list) -> set:
    entities = set()
    ent_type, start = None, None

    for i, tag in enumerate(list(tags) + ["O"]):  # "O" cuối để đóng thực thể còn mở
        is_begin = tag.startswith("B-")
        is_inside = tag.startswith("I-")

        # kết thúc thực thể hiện tại
        if ent_type is not None and (is_begin or not is_inside or tag[2:] != ent_type):
            entities.add((ent_type, start, i - 1))
            ent_type = None

        # bắt đầu thực thể mới (I- lạc loài cũng được coi là bắt đầu thực thể mới)
        if ent_type is None and (is_begin or is_inside):
            ent_type, start = tag[2:], i

    return entities


def f1_ner(gts: list, pds: list) -> dict:
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for gt_tags, pd_tags in zip(gts, pds):
        gt_ents = get_entities(gt_tags)
        pd_ents = get_entities(pd_tags)

        for ent in pd_ents & gt_ents:
            tp[ent[0]] += 1
        for ent in pd_ents - gt_ents:
            fp[ent[0]] += 1
        for ent in gt_ents - pd_ents:
            fn[ent[0]] += 1

    def prf(t, p, n):
        precision = t / (t + p) if t + p > 0 else 0.0
        recall = t / (t + n) if t + n > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        return precision, recall, f1

    per_type = {}
    for ent_type in sorted(set(tp) | set(fp) | set(fn)):
        p, r, f = prf(tp[ent_type], fp[ent_type], fn[ent_type])
        per_type[ent_type] = {"precision": p, "recall": r, "f1": f}

    precision, recall, f1 = prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_type": per_type,
    }
