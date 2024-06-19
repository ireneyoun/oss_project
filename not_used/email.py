# 파이썬으로 이메일 보내는 코드
# 이상 행동별로 답변 다르게 수정할 예정

import smtplib
from email.mime.text import MIMEText

smtp = smtplib.SMTP('smtp.gmail.com', 587)

smtp.ehlo()

smtp.starttls()

smtp.login('sunyoungju517@gmail.com', 'scfz vnwo qpud vrbv')

msg = MIMEText('내용 : 이상 행동 감지')
msg['Subject'] = '제목: 이상 행동 감지'

smtp.sendmail('sunyoungju517@gmail.com', 'sminji721@gmail.com', msg.as_string())

# 첫 번째 이메일 : 발신자, 두 번째 이메일 : 수신

smtp.quit()
