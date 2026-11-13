"""Export a fitted synthetic model for the browser what-if workbench."""
import json
from pathlib import Path
from opti.data import SyntheticDataConfig, generate_synthetic_logs
from opti.features import FeatureBuilder
from opti.model import OptiModel

def export(path):
    cfg=SyntheticDataConfig(n_days=210,seed=42)
    df,_=generate_synthetic_logs(cfg)
    fb=FeatureBuilder().fit(df.iloc[:160]);X,y=fb.transform_with_target(df.iloc[:160]);model=OptiModel().fit(X,y)
    testX,testY=fb.transform_with_target(df.iloc[157:])
    today=df.iloc[-1][fb.behavior_cols]
    baseline=fb.transform_single_with_history(today,df.iloc[:-1].tail(max(fb.lags)))
    result={'source':'Synthetic data; not personal predictions or causal effects','seed':42,'training_rows':len(X),'heldout_rows':len(testX),
      'columns':fb.behavior_cols,'means':fb.means_.to_dict(),'stds':fb.stds_.to_dict(),'features':fb.feature_names_,
      'weights':model.coef_.tolist(),'intercept':model.intercept_.tolist(),'targets':model.target_names_,
      'baseline':baseline.tolist(),'baseline_prediction':model.predict(baseline.reshape(1,-1)).tolist()[0],'current':today.to_dict(),'bounds':cfg.bounds,'interactions':fb.interactions,
      'heldout_r2':model.score(testX,testY),'alpha':model.alpha_}
    Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',default='demo/model.json');a=p.parse_args();export(a.output)
