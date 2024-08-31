TOTAL_PROMPT = """<대화>
{dialog}

<참고>
{ref}

<문제>
위 대화에서 {category}에 해당하는 것으로 맞는 선택지는? ({category}의 정의는 {def}이다.)

<보기>
A. {inference_1}
B. {inference_2}
C. {inference_3}

<답>
{answer}"""


INPUT_PROMPT = """<대화>
{dialog}

<참고>
{ref}

<문제>
위 대화에서 {category}에 해당하는 것으로 맞는 선택지는? ({category}의 정의는 {def}이다.)

<보기>
A. {inference_1}
B. {inference_2}
C. {inference_3}

<답>
"""