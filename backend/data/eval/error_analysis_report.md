# CLIR System Error Analysis Report

Generated: 2026-01-23 00:24:23

---

## Summary

Total findings: 2

- **Named Entity Mismatch**: 2 case(s)

---

## Named Entity Mismatch

### Case Study 1

**Query**: `news about Dhaka`

- **Detected Entities**: Dhaka

**Retrieved Documents (Top 5)**:

1. **কেরানীগঞ্জে বহুতল ভবনে আগুন, নিয়ন্ত্রণে কাজ করছে ১৪ ইউনিট**
   - URL: https://www.dhakapost.com/national/416253
   - Keywords: ঢাকার, খবর, ঢাকা
2. **ভূমিকম্পে ঢাকার অন্তত চার ভবনে ধস-ফাটল, হতাহতের শঙ্কা**
   - URL: https://www.dhakapost.com/national/411136
   - Keywords: ঢাকার, খবর, ঢাকা
3. **ঢাকার যে ৬ জায়গায় গেলে পাবেন অথেনটিক নেহারি**
   - URL: https://www.haal.fashion/food/4idfj5weyn
   - Keywords: ঢাকার, খবর, ঢাকা
4. **মেস থেকে জগন্নাথ বিশ্ববিদ্যালয়ের শিক্ষার্থীর ঝুলন্ত মরদেহ উদ্ধার**
   - URL: https://www.prothomalo.com/bangladesh/w0luthx6ux
   - Keywords: ঢাকার, খবর, ঢাকা
5. **সাভারে পার্কিং করা বাসে আগুন**
   - URL: https://www.dhakapost.com/country/416429
   - Keywords: খবর, ঢাকা

**Analysis**:

Query contains named entities: Dhaka, but only 0/5 top results contain these entities. This suggests NER mapping or entity matching may be failing.

**Recommendation**:

Improve named entity mapping between languages. Consider using a more comprehensive entity dictionary or entity-aware retrieval boosting.

---

### Case Study 2

**Query**: `Bangladesh economy`

- **Detected Entities**: Bangladesh

**Retrieved Documents (Top 5)**:

1. **ঢাকা কলেজের অর্থনীতি বিভাগের ১৮টি ব্যাচের পুনর্মিলনী**
   - URL: https://www.dhakapost.com/campus/101993
   - Keywords: বাংলাদেশের, অর্থনীতি, বাংলাদেশ
2. **ব্যবসা এবং অর্থনীতির উন্নয়নে খালেদা জিয়ার অবদান সুদূরপ্রসারী: সিমিন রহমান**
   - URL: https://www.prothomalo.com/politics/7s1nxf2qgt
   - Keywords: বাংলাদেশের, অর্থনীতি, বাংলাদেশ
3. **শুধু সুষ্ঠু নির্বাচনই নয়, অর্থনৈতিক শৃঙ্খলাও ফেরাতে হবে**
   - URL: https://www.prothomalo.com/opinion/column/hhi6dfbooe
   - Keywords: বাংলাদেশের, অর্থনীতি, বাংলাদেশ
4. **সামুদ্রিক মৎস্যসম্পদ সুনীল অর্থনীতিকে সমৃদ্ধ করবে**
   - URL: https://www.dhakapost.com/agriculture-and-nature/58524
   - Keywords: বাংলাদেশের, অর্থনীতি, বাংলাদেশ
5. **বৈজ্ঞানিকভাবে মাছ চাষ খাদ্য নিরাপত্তায় নতুন সম্ভাবনা সৃষ্টি করছে**
   - URL: https://www.dhakapost.com/national/416707
   - Keywords: বাংলাদেশের, অর্থনীতি, বাংলাদেশ

**Analysis**:

Query contains named entities: Bangladesh, but only 0/5 top results contain these entities. This suggests NER mapping or entity matching may be failing.

**Recommendation**:

Improve named entity mapping between languages. Consider using a more comprehensive entity dictionary or entity-aware retrieval boosting.

---

