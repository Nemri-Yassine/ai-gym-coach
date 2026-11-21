from awc.coach.policy import Policy


def test_policy_priorities():
    p = Policy(good_every=3)
    d = p.decide(risk=0.8, form_mistake=2, form_prob=0.9)
    assert d.type == "stop"
    p2 = Policy(good_every=3)
    d = p2.decide(risk=0.5, form_mistake=2, form_prob=0.9)
    assert d.type == "slow"
    p3 = Policy(good_every=3)
    d = p3.decide(risk=0.1, form_mistake=2, form_prob=0.9)
    assert d.type == "form"
    # one message per call
    assert d.message


def test_policy_debounce():
    p = Policy(good_every=1)
    a = p.decide(risk=0.8)
    b = p.decide(risk=0.8)
    assert a.type == "stop"
    assert b.type == "none"  # same message suppressed
