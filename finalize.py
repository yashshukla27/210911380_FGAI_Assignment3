"""Generate a personalized submission and technical report from saved evidence."""
import argparse
import csv
import json
import re
from pathlib import Path
from xml.sax.saxutils import escape
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.oxml.ns import qn
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle
from src.common import read,dump

NAME='Yash Shukla'
REG='210911380'
TITLE='English to Hindi Machine Translation with Four Transformer Models'
DATASET='https://huggingface.co/datasets/cfilt/iitb-english-hindi'


def read_csv(p):
    with p.open(encoding='utf-8-sig') as f: return list(csv.DictReader(f))


def review_summary(root,models):
    comments=[]
    finished=True
    for name in models:
        p=root/'results'/f'{name}_human_review.csv'
        q=root/'results'/f'{name}_ethical_probes.csv'
        if not p.exists() or not q.exists():
            finished=False
            continue
        rows,probes=read_csv(p),read_csv(q)
        fields=['adequacy_0_to_2','bias_0_to_2','hallucination_0_to_2','culture_0_to_2','sensitive_content_0_to_2']
        rated=[r for r in rows if all(r[k] in ['0','1','2'] for k in fields) and r['reviewer_notes'].strip()]
        first_ids={r['id'] for r in rows[:10]}
        enough=first_ids.issubset({r['id'] for r in rated}) and len(rated)>=min(10,len(rows))
        probe_done=all(r['human_severity_0_to_2'] in ['0','1','2'] and r['human_comment'].strip() for r in probes)
        finished=finished and enough and probe_done
        comments.append(f'{name}: {len(rated)} test translations rated; {sum(bool(r["human_comment"].strip()) for r in probes)} probe comments recorded.')
        for r in rated[:2]: comments.append(name+' test evidence: '+r['reviewer_notes'])
        for r in probes[:2]:
            if r['human_comment'].strip(): comments.append(name+' '+r['category']+': '+r['human_comment'])
    return finished,comments


def style_doc(doc):
    for style in doc.styles:
        for border in list(style.element.iter(qn('w:pBdr'))):
            border.getparent().remove(border)
    sec=doc.sections[0]
    sec.top_margin=sec.bottom_margin=Inches(.72)
    sec.left_margin=sec.right_margin=Inches(.8)
    normal=doc.styles['Normal']
    normal.font.name='Calibri'
    normal.font.size=Pt(10.5)
    normal.paragraph_format.space_after=Pt(7)
    for n in ['Title','Heading 1','Heading 2']:
        doc.styles[n].font.color.rgb=RGBColor(0,0,0)
    doc.styles['Title'].font.size=Pt(25)
    doc.styles['Heading 1'].font.size=Pt(15)
    doc.styles['Heading 1'].paragraph_format.space_before=Pt(12)
    doc.styles['Heading 1'].paragraph_format.space_after=Pt(6)
    doc.core_properties.author=NAME


def add_table(doc,headers,rows):
    t=doc.add_table(rows=1, cols=len(headers))
    t.style='Light Shading Accent 1'
    for c,v in zip(t.rows[0].cells,headers):c.text=str(v)
    for row in rows:
        for c,v in zip(t.add_row().cells,row):c.text=str(v)
    return t


def make_submission(root,repo,ready):
    folder=root/'submission'
    folder.mkdir(exist_ok=True)
    stem=f'Yash_Shukla_{REG}_FGAI_Assignment3'
    status='Ready for student review and LMS upload' if ready else 'DRAFT - model runs, output review and repository link must be completed before LMS upload'
    pairs=[('Student name',NAME),('Registration number',REG),('Course','Foundations of Generative AI (ICT4403)'),
           ('Assignment title','Assignment 3 - Comparative Machine Translation'),
           ('Developed autoregressive system',TITLE),('Language direction','English to Hindi'),
           ('Dataset','IIT Bombay English-Hindi Parallel Corpus'),('Dataset link',DATASET),
           ('GitHub repository',repo or 'PENDING - supply the published repository URL'),('Status',status)]
    doc=Document();style_doc(doc)
    doc.add_paragraph('Foundations of Generative AI',style='Title')
    doc.add_paragraph('Assignment 3 submission details',style='Subtitle')
    for key,value in pairs:
        p=doc.add_paragraph();p.add_run(key+'\n').bold=True;p.add_run(value)
    doc.add_paragraph('The repository should contain the implementation, measured results, plots, generated translations, responsible-AI analysis and execution instructions.')
    doc.save(folder/(stem+'.docx'))
    # A portable equivalent PDF; Word can also export the DOCX through Save As PDF.
    styles=getSampleStyleSheet()
    styles['Normal'].fontSize=10
    styles['Normal'].leading=14
    story=[Paragraph('Foundations of Generative AI',styles['Title']),Paragraph('Assignment 3 submission details',styles['Heading2']),Spacer(1,12)]
    for key,value in pairs:
        story.extend([Paragraph('<b>'+escape(key)+'</b>',styles['Normal']),Paragraph(escape(value),styles['Normal']),Spacer(1,9)])
    SimpleDocTemplate(str(folder/(stem+'.pdf')),leftMargin=55,rightMargin=55,topMargin=50,bottomMargin=50).build(story)
    return folder


def make_report(root,metrics,ready,comments):
    cfg=read(root/'config.json') if (root/'config.json').exists() else read(root/'results/run_config.json')
    manifest=read(root/'results/data_manifest.json') if (root/'results/data_manifest.json').exists() else None
    doc=Document();style_doc(doc)
    doc.add_paragraph('English to Hindi Machine Translation',style='Title')
    doc.add_paragraph('Comparative implementation and evaluation report',style='Subtitle')
    doc.add_paragraph(f'{NAME} | Registration {REG}\nFoundations of Generative AI ICT4403 | Assignment 3')
    doc.add_heading('Experiment status',1)
    if ready:doc.add_paragraph('All four model runs and the required output review are recorded. The following analysis reports this experiment only.')
    else:doc.add_paragraph('The implementation and methodology are provided below. The results section reports only executed models. A four-model conclusion requires all runs and review of the actual translations.')
    sections=[
    ('Objective','Compare a basic Transformer, mT5, LLaMA and Gemma on English-to-Hindi translation using a common test set. Assess translation quality, task adaptation cost, inference time, parameter count and responsible-AI risks. The decoder-only systems use autoregressive generation, satisfying the submission sheet wording as well as the translation task.'),
    ('Dataset and preprocessing',f'The IIT Bombay English-Hindi Parallel Corpus is a public parallel-text dataset. Published metadata lists 1,659,083 training, 520 validation and 2,507 test pairs. This implementation selects {cfg["train_size"]} training, {cfg["validation_size"]} validation and {cfg["test_size"]} test pairs from the corresponding official partitions. Source: {DATASET}'),
    ('Data separation','Normalize Unicode to NFC and collapse whitespace while preserving Hindi script, case and punctuation. Remove empty pairs, identical source/target pairs, text over 240 characters or 40 words, and sentences lacking the expected scripts. Remove repeated English or Hindi sentences across partitions, retaining test before validation before training. The training pool is a bounded buffered shuffle of 50,000 examples; this is a resource-limited short-sentence sample, not a uniform full-corpus sample. Train the basic tokenizer only on training pairs. Record ordered test IDs and a dataset revision hash.'),
    ('Basic Transformer','A randomly initialized PyTorch Transformer has separate encoder and decoder stacks, sinusoidal positional encoding, padding masks, causal decoder self-attention and encoder-decoder cross-attention. The decoder predicts the next Hindi token from previous Hindi tokens and encoded English text. Joint BPE has up to 8,000 tokens, with PAD, UNK, BOS and EOS reserved. The model uses two layers per stack, dimension 128, four attention heads, feed-forward dimension 512 and dropout 0.1. Train for six epochs with AdamW at 0.0003, batch size 32, gradient clipping 1.0 and label smoothing 0.1; retain the lowest validation NLL checkpoint.'),
    ('mT5 fine tuning','Use google/mt5-small, a pretrained multilingual encoder-decoder. Its pretraining alone is not task-specific translation training. Prefix each source with translate English to Hindi and train on the Hindi target with padding labels masked as -100. Full fine-tuning uses Adafactor at 0.0005 for two epochs, microbatch size one and gradient accumulation sixteen. Float32 and gradient checkpointing avoid unstable float16 training and reduce activation memory. Select the checkpoint with lowest validation NLL. No validation or test pair enters the training objective.'),
    ('LLaMA and Gemma prompting','Use meta-llama/Llama-3.2-1B-Instruct and google/gemma-3-1b-it. Prompting is explicitly allowed by the assignment. Both receive the same translation instruction and two short demonstrations drawn only from training data, wrapped in their native chat templates. Request only the Hindi translation and preservation of names, quantities and negation. Remove prompt tokens before decoding generated output. No weight updates occur; adaptation time is therefore not applicable rather than a measured zero.'),
    ('Comparable decoding and measurement','Every model translates the same ordered test sentences with greedy decoding, one beam, batch size one and a maximum of 128 new tokens. Full inputs are retained; overlength inputs raise an error instead of silently truncating. Record outputs that hit the generation cap. Equal token limits do not imply equal text length because tokenizers differ. A validation sentence warms up each model. Synchronize CUDA before and after timing and include input tokenization, generation and text decoding. Count parameters from the loaded model and record float precision, model revision, hardware and software. Training time includes validation and checkpoint writes but excludes pretraining, dataset download and setup.'),
    ('Evaluation metrics','Corpus BLEU uses SacreBLEU with the 13a tokenizer and one Hindi reference. chrF uses character n-grams through order six and beta two, with no word n-grams. Both are reported on a 0 to 100 scale. chrF is the primary ranking metric, with BLEU as a secondary measure. Neither metric directly proves semantic faithfulness, fairness or safety. Hindi inflection and valid alternative phrasing can lower reference overlap. Preserve metric signatures so the calculation can be reproduced.'),
    ('Architecture comparison','A basic encoder-decoder can use a compact vocabulary and a much smaller parameter count but must learn translation from a limited corpus. mT5 transfers multilingual pretraining and then learns the translation task. The prompted causal models avoid task training, but their larger weights and prompt processing still cost memory and inference time. A measured difference is jointly affected by size, pretraining, vocabulary, instruction tuning, adaptation and data budget. This experiment cannot isolate architecture as the causal explanation.'),
    ('Temperature experiment','Keep the main benchmark greedy. On the same first 25 test examples, evaluate temperatures 0.3, 0.7 and 1.0 with seeds 42, 43 and 44. Disable top-k and nucleus filtering for sampled runs so temperature alone controls the distribution. Save translations, per-seed scores, means and standard deviations. Compare faithfulness as well as BLEU and chrF. Treat results as a descriptive sensitivity analysis, without choosing the main decoding settings using test scores.'),
    ('Responsible AI analysis','Review actual outputs for unsupported gender or identity assumptions, invented entities and dates, omitted negation, cultural/idiomatic distortion and changes to sensitive warnings. Eight purpose-built probes include paired doctor/nurse sentences, a medical negation, a meeting date, cultural terms, an idiom, an embedded instruction and an inclusive identity statement. These illustrative probes are separate from the scored benchmark. Heuristic digit changes, empty output, missing Devanagari and length-cap hits trigger review but do not prove ethical violations.'),
    ('Mitigation and limitations','Use balanced data, bilingual review, cultural/domain glossaries, source-delimiting instructions and checks for names, numbers, units and negation. Require qualified review of medical or legal translations and protect private text. Cite sources, respect dataset/model terms and exclude secrets and model weights from public uploads. The short-sentence sample and eight probes do not establish broad safety; longer text, dialects and other domains need further evaluation. Unknown pretrained-data overlap remains a contamination risk.')]
    for heading,body in sections:
        doc.add_heading(heading,1);doc.add_paragraph(body)
        if heading=='Dataset and preprocessing' and manifest:
            add_table(doc,['Partition','Examined','Eligible unique','Selected'],[[s,v['examined'],v['after_deduplication'],v['selected']] for s,v in manifest['splits'].items()])
            doc.add_paragraph('Dataset revision: '+manifest['revision'])
    doc.add_heading('Measured results',1)
    if metrics:
        add_table(doc,['Model','BLEU','chrF','Training s','Inference s'],[[r['model'],f"{r['bleu']:.2f}",f"{r['chrf']:.2f}",'N/A' if r['training_seconds'] is None else f"{r['training_seconds']:.1f}",f"{r['inference_seconds']:.1f}"] for r in metrics.values()])
        for r in metrics.values():
            doc.add_paragraph(f"{r['model']}: {r['parameters']:,} parameters; {r['test_sentences']} test sentences; {r['capped_outputs']} outputs reached the token cap.")
        for image in ['quality_comparison.png','compute_comparison.png','temperature_analysis.png']:
            p=root/'results'/image
            if p.exists():doc.add_picture(str(p),width=Inches(6.4))
    else:doc.add_paragraph('No completed model metrics are available. Scores, timings and rankings are not inferred or fabricated.')
    if 'basic' in metrics and metrics['basic']['bleu']==0 and metrics['basic']['chrf']==0:
        doc.add_paragraph('The basic model has zero BLEU and chrF in this run. Inspect the saved outputs before interpreting its timing. The supplied CPU baseline produced empty translations, so short inference time reflects early termination rather than useful translation efficiency.')
    doc.add_heading('Review of generated translations',1)
    for c in comments or ['Actual-output review remains outstanding. Complete the per-model human review and ethical probe files before final submission.']:
        doc.add_paragraph(c)
    doc.add_heading('Conclusion',1)
    if set(metrics)=={'basic','mt5','llama','gemma'}:
        best=max(metrics.values(),key=lambda r:(r['chrf'],r['bleu']))
        doc.add_paragraph(f"The highest chrF in this experiment is achieved by {best['model']}, with chrF {best['chrf']:.2f} and BLEU {best['bleu']:.2f}. This is a subset-specific empirical ranking, not a universal architecture claim. Consider inference time, memory and human review alongside quality.")
    else:doc.add_paragraph('An overall winner cannot be identified until all four models have produced translations and metrics on the same test set. The code produces a measured ranking after completion; the current evidence does not support one.')
    doc.add_heading('Observed baseline failure',1)
    if 'basic' in metrics and metrics['basic']['bleu']==0 and metrics['basic']['chrf']==0:
        doc.add_paragraph('In the supplied CPU run, every test translation and all eight probe translations were empty. For example, the source Do not take this medicine without consulting a doctor produced no Hindi text, omitting the entire warning. The doctor and nurse probes both produced no translation, so they provide no meaningful evidence about gender fairness. The prasad and puja probe also omitted all cultural meaning. These are observed omission failures, not invented example outputs.')
        doc.add_paragraph('Validation NLL was lowest at epoch one even as training loss fell later. This is consistent with limited learning and a possible domain mismatch between the small training pool and held-out sentences. The observation does not identify a single cause. Future changes should be selected on validation data and assessed with a fresh final test set if this test outcome influenced the redesign.')
    else:
        doc.add_paragraph('Use the actual per-model review notes above to explain any omissions, false additions or cultural changes observed in this run.')
    doc.add_heading('References',1)
    for s in ['Kunchukuttan et al. (2018), The IIT Bombay English-Hindi Parallel Corpus. https://aclanthology.org/L18-1548/',
              'Vaswani et al. (2017), Attention Is All You Need. https://arxiv.org/abs/1706.03762',
              'Xue et al. (2021), mT5. https://arxiv.org/abs/2010.11934',
              'Meta model card. https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct',
              'Google model card. https://huggingface.co/google/gemma-3-1b-it',
              'SacreBLEU. https://github.com/mjpost/sacrebleu',
              'Popovic (2015), chrF. https://aclanthology.org/W15-3049/']:
        doc.add_paragraph(s)
    doc.save(root/'submission/Technical_Report.docx')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--repo-url',default='');p.add_argument('--draft',action='store_true')
    a=p.parse_args();root=Path(a.root).resolve()
    repo_ok=bool(re.fullmatch(r'https://github\.com/[A-Za-z0-9-]+/[A-Za-z0-9_.-]+/?',a.repo_url))
    metrics=read(root/'results/metrics.json') if (root/'results/metrics.json').exists() else {}
    complete=set(metrics)=={'basic','mt5','llama','gemma'}
    reviewed,comments=review_summary(root,metrics)
    ready=complete and reviewed and repo_ok
    if not ready and not a.draft:
        raise SystemExit('Not ready: require all four model results, rated test/probe reviews and a valid repository URL. Use --draft only for a clearly marked preview.')
    make_submission(root,a.repo_url,ready)
    make_report(root,metrics,ready,comments)
    dump(root/'submission/readiness.json',{'four_models_complete':complete,'human_review_complete':reviewed,
        'repository_url_syntax_valid':repo_ok,'repository_url':a.repo_url,'ready_for_student_review':ready,
        'note':'Check repository accessibility and document layout before LMS upload. URL syntax is not proof of publication.'})
    print('Submission documents created:',root/'submission')

if __name__=='__main__':main()
