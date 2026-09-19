def participant_split(records, test_fraction=.2):
 ids=sorted({r['participant_id'] for r in records}); n=max(1,round(len(ids)*test_fraction)) if len(ids)>1 else 0; test=set(ids[-n:]) if n else set(); train=[r for r in records if r['participant_id'] not in test]; testing=[r for r in records if r['participant_id'] in test]; assert_no_leakage(train,testing); return train,testing
def assert_no_leakage(train,test):
 overlap={r['participant_id'] for r in train}&{r['participant_id'] for r in test}
 if overlap: raise ValueError('Participant leakage detected: '+str(sorted(overlap)))
