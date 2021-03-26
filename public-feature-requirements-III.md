# Public Feature Requirements III
## FS1201: CRIO Research
Do some ground research on CRI-o and summarize it.

Links:

​https://github.com/cri-o/cri-o​

​https://kubernetes.io/docs/setup/production-environment/container-runtimes/​

​https://kubernetes.io/docs/setup/production-environment/container-runtimes/#cri-o​

​https://github.com/kubernetes/community/blob/master/contributors/devel/sig-node/container-runtime-interface.md​

Container Runtimes: CRI Docker RKT 
https://coreos.com/rkt/​

What is POD? 
https://kubernetes.io/docs/concepts/workloads/pods/#what-is-a-pod​

## FS1202: Nemo Research

Do some library research on Nemo and write an article about your collection. You must have done at least 10 hour research to finish this task.

[Nemo: Data discovery at Facebook - Facebook Engineering](https://engineering.fb.com/2020/10/09/data-infrastructure/nemo/)
*Nemo allows engineers to quickly discover the information they need, with high confidence in the accuracy of the results.*\
*engineering.fb.com*
#

## FS1203: CG Research

Do some library research on CG/SQL and write a medium article about your collection. You must have done at least 10 hour research to finish this task.

[CG/SQL: Easy, accurate SQLite code generation - Facebook Engineering](https://engineering.fb.com/2020/10/08/open-source/cg-sql/)
*CG/SQL is a code generation system for the popular SQLite library that enables engineers to create complex stored procedures with very large queries.*\
*engineering.fb.com*
#

## FS1204: TactIndex

TactIndex: 
Stack Index + Other criteria

​https://landscape.cncf.io/category=cloud-native-storage&grouping=category 

The visual should be in this format
#

## FS1205: GishML

GishML: (Applicable for Wiki page) 
This command will check my previous changes in Wiki and suggest me the message info for Git
#

## FS1206: MemeNews

Convey News in comical ways:

Sample: Eyes on the Pfizer! always

Ideal:
https://likeshop.me/businessweek​
#

## FS1207: AutoCommentary

Cricket Commentary with NLP

Automate the Cricket commentary for each ball
#

## FS1208: CompareSub

Auto Subtitle vs Human Subtitle

Justin Bieber - Sorry (PURPOSE : The Movement) 
https://www.youtube.com/watch?v=fRh_vgS2dFE​

Check how much % the auto subtitle varies from human subtitle

Try with YT Subtittle, Google and AWS
#

## FS1209: SongCompare

Dua Lipa - New Rules (Official Music Video) 
https://www.youtube.com/watch?v=k2qgadSvNyU​

vs

Justin Bieber - Sorry (PURPOSE : The Movement) 
https://www.youtube.com/watch?v=fRh_vgS2dFE​

Compare these songs for the next 7 days

Timeseries: 
1. Every 15 mins collect the views by using Python 
2. Store them into any Timeseries DB by using Python 
3. Give me a report which Song would get more views

Tools/Libs:

Use Anaconda/Miniconda and Python 3.7+ environment

Use VSCode
#

## FS1210: RunLy

Collect song Lyrics from a running video (assuming the video has subtitle)

Use any video ML algorithm to get the subtitle
#

## FS1211: Controversary Meter

Controversary meter

[Opinion | Is There a Doctor in the White House? Not if You Need an M.D.](https://www.wsj.com/articles/is-there-a-doctor-in-the-white-house-not-if-you-need-an-m-d-11607727380)\
*Jill Biden should think about dropping the honorific, which feels fraudulent, even comic.*\
*www.wsj.com*

There are a lot of outcry on this topic like WSJ should not have diminsh Jill Biden like that. We have to measure topics like this and give a meter for these topics.

Sample:
https://www.elle.com/culture/career-politics/a34977519/michelle-obama-wsj-dr-jill-biden-op-ed-response/​
#

## FS1212: Mysogynist Meter

[Opinion | Is There a Doctor in the White House? Not if You Need an M.D.](https://www.wsj.com/articles/is-there-a-doctor-in-the-white-house-not-if-you-need-an-m-d-11607727380)
*Jill Biden should think about dropping the honorific, which feels fraudulent, even comic.*\
*www.wsj.com*

Misogynist meter

Measure the article whether it is too Misogynist or not with 1-10 meter.
#

## FS1213: Synthetic addresses

Generate fake Ontario address which is not available by searching online. But it should be legit when we enter address on some websites.
#

## FS1214: MarakoBO

Trendy Marketing content for LCBO products

https://www.lcbo.com/
www.lcbo.com
#

## FS1215: Shufflor

LIFX Color Shuffler:

Change LIFX based on the Spotify song

Get the dominant colors from the song/album 

Supply the colors to LIFX API

[Introduction · LIFX HTTP Remote Control API](https://api.developer.lifx.com/)
*Control your users LIFX bulbs remotely with the LIFX HTTP Remote Control API.*\
*api.developer.lifx.com*
#

## FS1216: Colorify

Get the dominant colors of a song in Spotify

[Web API Reference | Spotify for Developers](https://developer.spotify.com/documentation/web-api/reference/#endpoint-get-playlist-cover)
*Music, meet code. Powerful APIs, SDKs and widgets for simple and advanced applications.*\
*developer.spotify.com*

#

## FS1217: cURL to Postman Converter

cURL to Postman converter

    1    curl -X POST "https://api.lifx.com/v1/lights/all/effects/breathe" \
    2        -H "Authorization: Bearer YOUR_APP_TOKEN" \
    3        -d 'period=2' \
    4        -d 'cycles=5' \
    5        -d 'color=green'

Convert this to Postman API
#

## FS1218: Nayanthara Clock

Make Nayanthara clock like below
<center><img src=clock.png alt="alt text" width="400" height="whatever"></center>

Come up with your own creativity to show Nayanthara faces or movie characters to show Nayanthara in the clock.

Sample:

​https://featurepreneur.github.io/nayanthara-clock/​
#

## FS1219: Chemical Clock

Make a clock with Chemical element names to show your creativity.
#

## FS1220: MinIO

Do some research on MinIO and come up with a POC with it.

[MinIO | High Performance, Kubernetes Native Object Storage](https://min.io/)
*MinIO's High Performance Object Storage is Open Source, Amazon S3 compatible, Kubernetes Native and is designed for cloud native workloads like AI.*\
*min.io*
#

## FS1221: Average Pricing History

Archive Pricing history

My Netflix subscription changed from 12$CAD to 18$CAD over 7 years. I will have to create a simple app to collect these public info.

Netflix, Hulu, Amazon Video, Instacart, etc.
#

## FS1222: KYR Canada

Know Your Rights (KYR)

Create a simple ML Feed engine will send user some information daily about worker rights with fun content involved.

Target Audience: 
New Immigrant in Canada

We will create this as a plugin and try to set it through Featurepreneur
#

## FS1223: Blog

I need to setup my blogs like this

[AMQP, RabbitMQ and Celery - A Visual Guide For Dummies](https://www.abhishek-tiwari.com/amqp-rabbitmq-and-celery-a-visual-guide-for-dummies/)\
*Work in Progress  Celery is an asynchronous distributed task queue. RabbitMQ is a message broker which implements the Advanced Message Queuing Protocol (AMQP). Before we describe relationship between RabbitMQ and Celery, a quick overview of AMQP will be helpful [1][2].  Amqp Key Terms Message Or Task A message or*\
*www.abhishek-tiwari.com*

https://ghost.org/docs/install/docker/\
*ghost.org*

https://ghost.org/docs/install/\
*ghost.org*

[Docker Hub](https://hub.docker.com/_/ghost/)
*hub.docker.com*
#

## FS1224: Play with Docker

[Play with Docker](https://labs.play-with-docker.com/)\
labs.play-with-docker.com

Do some ground experiments on Play with Docker and summurize your experiments.
#

## FS1225: Compare TV Shows Books

Compare TV shows/books/movies like this
    
### JAVA:       

        Highcharts.chart('container', {

        chart: {
            polar: true,
            type: 'line'
        },

        accessibility: {
            description: ''
        },

        title: {
            text: 'Chennai 28 vs Chennai 28 II',
            x: -80
        },

        pane: {
            size: '80%'
        },

        xAxis: {
            categories: ['Fun', 'Thrilling', 'Easy Going', 'Songs', 'Fresh'],
            tickmarkPlacement: 'on',
            lineWidth: 0
        },

        yAxis: {
            gridLineInterpolation: 'polygon',
            lineWidth: 0,
            min: 0
        },

        tooltip: {
            shared: true,
            pointFormat: '<span style="color:{series.color}">{series.name}: <b>{point.y:,.0f}</b><br/>'
        },

        legend: {
            align: 'right',
            verticalAlign: 'middle',
            layout: 'vertical'
        },

        series: [{
            name: 'Chennai 28',
            data: [9, 4, 8, 9, 10],
            pointPlacement: 'on'
        }, {
            name: 'Chennai 28 II',
            data: [4, 8, 4, 5, 4],
            pointPlacement: 'on'
        }],

        responsive: {
            rules: [{
                condition: {
                    maxWidth: 500
                },
                chartOptions: {
                    legend: {
                        align: 'center',
                        verticalAlign: 'bottom',
                        layout: 'horizontal'
                    },
                    pane: {
                        size: '70%'
                    }
                }
            }]
        }

    });
#

## FS1226: Data Collector CLI

Do some ground work on PM2

Write a scheduler in PM2 to collect data regularly by calling Python script.
#

## FS1227: HTML Mis-rendering

[Lenovo ThinkPad T450 14" Ultrabook (Intel Core i5-4300U / 8GB RAM / 256GB SSD) - Certified Refurbished - 1 Year Warranty | Best Buy Canada](https://www.bestbuy.ca/en-ca/product/lenovo-thinkpad-t450-14-ultrabook-intel-core-i5-4300u-8gb-ram-256gb-ssd-certified-refurbished-1-year-warranty/14340739)
*Slim and portable at 0.83" thin, the 14" ThinkPad T450 Ultrabook from Lenovo is built with ports and connectivity options that are often found on mobile workstations. Designed for business productivity, this system is powered by a 1.9 GHz Intel Core i5-4300U processor that allows you to run multiple applications simultaneously. If you need more power, the processor can be overclocked to 2.9 GHz and the system's 8GB of RAM allows the computer to quickly access frequently-used files and programs.*
*www.bestbuy.ca*

[BestBuy HTML Issue.pdf - 3MB](BestBuyHTMLIssue.pdf)

If you see the rendered html, it has issue like 

    1    1 x&nbsp;10/100/1000 Mb/s Gigabit Ethernet (RJ45)
    2    1 x&nbsp;3.5 mm Headphone/Microphone Combo Jack
We have to analyze how many pages it is rendered like that and then make a report to them.
#

## FS1228: HTML Mis-rendering

HMTL Mis-rendering archive

Archive HTML mis-rendering pages like in FS1227 and then store them in the DB. Show a report with Flask and Jinja.
#

## FS1229: Password Validator
<center><img src=pswd.png alt="alt text" width="400" height="whatever"></center>
Password
We need to replicate this password checking as a small UI page.

#

## FS1230: Error Collector
<center><img src=error.png alt="alt text" width="400" height="150"></center>
<center>Best Buy Critical Error</center>
We have to save errors like this and predict the revenue drop for Best Buy like companies.

#

## FS1231: Before ML

We need to collect various ML candidates' titles to understand how did they migrate from other industry to ML.

https://www.linkedin.com/in/quantscientist/\
www.linkedin.com
#

## FS1232: TactML Score
You can check TactML Score by clicking on the image below which would navigate you to our YouTube Channel:

[![TactML Score](https://img.youtube.com/vi/2RSzkFw4LFs/hqdefault.jpg)](https://www.youtube.com/watch?v=2RSzkFw4LFs "TactML")

We have come up with TactML Score to identify researchers like Yoshua Bengio\

Criteria:\
Independent Research\
Visionary Meter
#

## FS1233: Tech or Not
You can check Teach or Not by clicking on the image below which would navigate you to our YouTube Channel:

[![TactML Score](https://img.youtube.com/vi/2RSzkFw4LFs/hqdefault.jpg)](https://www.youtube.com/watch?v=2RSzkFw4LFs "TactML")
Classify tech or non-tech video among 10,000 videos.
#

## FS1234: AI Info Maker
AI Info maker

You can check AI Info Maker by clicking on the image below which would navigate you to our YouTube Channel:\
[![AI Info Maker](https://img.youtube.com/vi/_UUcGedcqSY/hqdefault.jpg)](https://www.youtube.com/watch?v=_UUcGedcqSY "AI Info Maker")]\

When we play this video, it shoud show 

"Harry Shum might have graduated his school in 1980s"

"HS might have gone to USA for AI learning in 1983"

Everything must be based on the video content and we should not search things online.
#

## FS1235: Video2Text
[![Video2Text](https://img.youtube.com/vi/uawLjkSI7Mo/hqdefault.jpg)](https://www.youtube.com/watch?v=uawLjkSI7Mo "Video2Text")

You need to convert this video to text with Amazon, Google, MS AI tools and do a benchmarking which one is more accurate.

You can start with a sample 2 mins audio as a sample work.
#

## FS1236: Collect 200 Research Papers

Collect 200 research papers
Topics
1. ML
2. NLP
3. DL
All these papers should be in PDF format
Sources:
1. Neurips 
2. Research gate 
3. Google scholar
4. Arxiv
#

## FS1237: NA ML Researchers

Find 300 North American ML Researchers

Data Collection
#

## FS1238: Random Foods

Get random food images and convert it as an API:

[Random foods generator - GeneratorMix](https://www.generatormix.com/random-foods-generator)\
*www.generatormix.com*

#

## FS1239: Featurepreneur Analytics

Featurepreneur Analytics have to created
#

## FS1240: Random Avatar Links API

​https://avatars.alphacoders.com/avatars/random​

Get random avatar links as an API. 
#

## FS1241: Random Avatars

Classify random avatars as male or female

​https://avatars.alphacoders.com/avatars/random​
#

## FS1242: Country & Speed 

Speed - Country - Years

Find the historical numbers of vehicles' speed limit in various countries from the day vehicles introduced.

Show a graph of them
#

## FS1243: Font Installer on Mac via Python

​https://github.com/powerline/fonts​

Write  a simple script to clone (or download) and then install in Mac via console.
#

## FS1244: Screenshot by Python
[How can I take a screenshot/image of a website using Python?](https://stackoverflow.com/questions/1197172/how-can-i-take-a-screenshot-image-of-a-website-using-python)\
*What I want to achieve is to get a website screenshot from any website in python.*\
*stackoverflow.com*

[peterdalle/screenshot](https://github.com/peterdalle/screenshot)\
*Take a screenshot of a whole web page (including the page below the fold and dynamically loaded images) - peterdalle/screenshot*\
*github.com*

[ronnyml/python-screenshot-generator](https://github.com/ronnyml/python-screenshot-generator)\
*App to generate a screenshot from websites built with Python/Django and Selenium. - ronnyml/python-screenshot-generator*\
*github.com*

    #!/usr/bin/python
    # -*- coding: utf-8 -*-

    import uuid
    import time
    import optparse
    import json
    import urllib2
    import os
    import logging

    from selenium import webdriver
    from pyvirtualdisplay import Display
    from PIL import Image


    SAVE_PATH = "img"
    DRIVERS = ["firefox", "phantom"]
    DRIVER = "firefox"
    WAIT_INTERVAL = 2
    URL = None
    BROWSER = None
    IMAGE_QUALITY = 60

    def init():
        global URL, DRIVER, DRIVERS, WAIT_INTERVAL, SAVE_PATH, IMAGE_QUALITY
        parser = optparse.OptionParser(usage=" ")

        parser.add_option("-u", "--url", dest="url", metavar="http://www.google.com")
        parser.add_option("-d", "--driver", dest="driver", metavar="firefox or phantom")
        parser.add_option("-i", "--interval", dest="interval", metavar="2")
        parser.add_option("-p", "--path", dest="save_path", metavar="img")
        parser.add_option("-q", "--quality", dest="quality", metavar="90")

        (options, args) = parser.parse_args()

        if options.url is not None:
            URL = options.url
        else:
            parser.print_help()
            exit()

        if options.driver is not None:
            if options.driver in DRIVERS:
                DRIVER = options.driver
            else:
                parser.print_help()
                exit()

        if options.interval is not None:
            try:
                WAIT_INTERVAL = float(options.interval)
            except Exception, msg:
                pass

        if options.save_path is not None:
            SAVE_PATH = options.save_path

        if options.quality is not None:
            if options.quality.isdigit():
                quality = int(options.quality)
                if quality < 1:
                    quality = 1

                if quality > 100:
                    quality = 100

                IMAGE_QUALITY = quality

        if not os.path.exists(SAVE_PATH):
            try:
                os.makedirs(SAVE_PATH)
            except Exception, msg:
                message("error", str(msg))
                exit()


    def message(status, message):
        data = {"status": status, "message": message}
        print json.dumps(data)


    def create_display():
        try:
            display = Display(visible=0, size=(1280, 1024))
            display.start()
            logging.info("Virtual Display Started")
        except:
            pass


    def create_browser():
        global BROWSER

        if DRIVER == "firefox":
            try:
                BROWSER = webdriver.Firefox()
            except Exception, msg:
                message("error", str(msg))
                exit()

        if DRIVER == "phantom":
            try:
                BROWSER = webdriver.PhantomJS()
            except Exception, msg:
                message("error", str(msg))
                exit()


    def check_url():
        global URL
        try:
            req = urllib2.urlopen(URL, timeout=20)
            if req.getcode() != 200:
                message("error", "url could not open")
                exit()
        except Exception, msg:
            message("error", str(msg))
            exit()


    def take_screenshot():
        global URL, BROWSER, SAVE_PATH, WAIT_INTERVAL, IMAGE_QUALITY

        SAVE_FILE_TEMP = "%s/%s_tmp.png" % (SAVE_PATH, uuid.uuid1())
        SAVE_FILE = "%s/%s.jpg" % (SAVE_PATH, uuid.uuid1())
        
        try:
            BROWSER.get(URL)

            body = BROWSER.find_element_by_css_selector('body')
            body.click()

            time.sleep(WAIT_INTERVAL)

            BROWSER.get_screenshot_as_file(SAVE_FILE_TEMP)

            BROWSER.close()

            im = Image.open(SAVE_FILE_TEMP)

            im.save(SAVE_FILE, "JPEG", quality=IMAGE_QUALITY)

            os.remove(SAVE_FILE_TEMP)

        except Exception, msg:
            message("error", str(msg))
            exit()

        message("ok", SAVE_FILE)

    if __name__ == "__main__":
        init()
        check_url()
        create_display()
        create_browser()
        take_screenshot()

[vladocar/screenshoteer](https://github.com/vladocar/screenshoteer)\
*Make website screenshots and mobile emulations from the command line. - vladocar/screenshoteer*\
*github.com*
​

Using Python capture the website page
#

## FS1245: Find Architecture Diagrams in Videos


<center><img src=arch.png alt="alt text" width="400" height="whatever"> </center>   
Find architecture diagrams like above from tech videos


Sample Video\
(Note: Click on the image which would navigate you to YouTube)
[![Sample Videos](https://img.youtube.com/vi/H2a0bwK-7es/hqdefault.jpg)](https://www.youtube.com/watch?v=H2a0bwK-7es "Sample Videos")
#

## FS1246: Aggregate Efficiency calculator

Calculate various countries' aggreate efficiency and create a visual with them.

Aggregate Efficiency: 
USA got 3%
Germany got 18% 
Japan got 21%

Source:
(Note: Click on the image which would navigate you to YouTube)
[![Aggregate Efficiency calculator](https://img.youtube.com/vi/QX3M8Ka9vUA/hqdefault.jpg)](https://www.youtube.com/watch?v=QX3M8Ka9vUA "Aggregate Efficiency calculator")

#

## FS1247: TubeLines

Create 3-7 lines about each tech video.
Try to automate with NLP highlighter.
Measure the accuracy
#

## FS1248: USA Double Standards

Show visual of USA double standards:

Sample source video\
https://www.youtube.com/watch?v=6ZH0bwUuT_A​
#

## FS1249: QGIS

Do a sample work with QGIS

QGIS tutorial:

[QGIS Tutorials and Tips — QGIS Tutorials and Tips](https://www.qgistutorials.com/en/)\
*www.qgistutorials.com*
#

## FS1250: Support Isabella

Support Isabella 
https://www.youtube.com/watch?v=7GkFzcUJTk8​

Do some visual based on Isabella's story.
#

## FS1251: Public Validator

Need to come up with a simple PublicValidator page

Validators neeed:

GitHub secrets
​https://docs.github.com/en/free-pro-team@latest/actions/reference/encrypted-secrets​

<center><img src=secret.png alt="alt text" width="400" height="whatever"> </center>
<center>github secrets</center>

TD Password

Anonymous Site
<center><img src=ano.png alt="alt text" width="400" height="whatever"> </center>
<center>anonymous site</center>
Adobe Password
<center><img src=adobe.png alt="alt text" width="400" height="whatever"> </center>
<center>Adobe password</center>

#

## FS1252: TACT ECS Chatbot

Tact ECS Chatbot

You have to create a Chatbot for ECS. Whoever wants to know about ECS, they can learn from this chatbot.

[What is Amazon Elastic Container Service? - Amazon Elastic Container Service](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/Welcome.html)\
*Run, stop, and manage Docker containers on a cluster using the Amazon Elastic Container Service Developer Guide.*\
*docs.aws.amazon.com*
#

## FS1253 : Handwritten Testimonials

official names: ClassicTesti

    1    ClassicTesti.org
    2    ClassicTesti.com
    3    ​
    4    are available
#    

## FS1254: Find Marketing Content

    1    The 3.0 release of JupyterLab brings many new features! Like a visual debugger
    2    $ 𝚙𝚒𝚙 𝚒𝚗𝚜𝚝𝚊𝚕𝚕 𝚓𝚞𝚙𝚢𝚝𝚎𝚛𝚕𝚊𝚋==3
    3    Features • Stepping into Python code in JupyterLab with the visual debugger • The table of contents extension now ships with JupyterLab. This makes it easy to see and navigate the structure of a document • Support for multiple display languages
    4    And many more check it here 
    5    https://lnkd.in/gk-Z4gg
    6    ​
    7    ⚡ Spread the Open Source love If you know an amazing project, paper or library drop me a message here on LinkedIn or Twitter @philipvollet 
    8    https://lnkd.in/gG3BgzG
    9
​    

In the above content, find the marketing content and highlight them by using NLP.
#

## FS1255: ReleaseKin

Show a table/visual for various library versions and their release date

[Compose file version 3 reference](https://docs.docker.com/compose/compose-file/compose-file-v3/)\
*Compose file reference*\
*docs.docker.com*
#

## FS1256: Google Form ++ 

It's hard to view the Google Form without navigation. If you could improve the viewing option by Left, Right Nav it would be great. You will have to read the Excel sheet in Python and show the viewing page
#

## FS1257: Open Source Contribution per capita

We need to find how many open source contributors in South India and Ontario and compare them regularly. 

We need to increate the open source contributors count in South India. This is the ultimate goal of this project.
#

## FS1258: California Exodus

Do some visual about this topic.
[![California Exodus](https://img.youtube.com/vi/&feature=emb_imp_woyt/hqdefault.jpg)](https://www.youtube.com/watch?v=&feature=emb_imp_woyt "California Exodus")

#

## FS1259:

Make a visual as this image.
<center><img src=chart.png alt="alt text" width="400" height="whatever"></center>

#

## FS1260:
Do some ground work on TinkerBell and write an article about it.

[Tinkerbell Docs - Home](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/Welcome.html)\
*Designed for scalable provisioning of physical hardware, Tinkerbell is both open-source and cloud native. Contributed by Equinix Metal.*
*docs.tinkerbell.org*
#

## FS1261:

Encoption - TBD
#

## FS1262:

LibAlerts - TBD
#

## FS1263: FakeProfileFinder

Be careful out there: LinkedIn is infested with fake profiles. Out of thousands of invites I've received, hundreds have been fake. How can you recognize a fake profile?

Fakers use patterns and make mistakes that give them out. Here are some examples:

1. You can't find the same person elsewhere with a simple Google search.

2. Their work and education history is from large institutions with no unique details.

3. The name and professional summary are generic.

4. Participated groups and pages don't show a unique history.

5. Historical participation on posts doesn't exist or doesn't fit the background or location.

6. The friend list doesn't fit the background.

7. With Google image search the picture is elsewhere with another name. This is less useful today because it is so easy to generate deepfake images.

The image below shows a fake profile that has gathered 413 connections, many of them Finns. The name is generic, lacking surname. The photo looks like an image bank model or a deepfake. The study program she claims to participate doesn't exist.

Have you recognized fake profiles among your invites yet? Any other ways you have used to recognize a fake profile?
<center><img src=lnkd.png alt="alt text" width="300" height="500">

sample fake profile</center>

source:

[Mikko Alasaarela on LinkedIn: ⚠️ Be careful out there: LinkedIn is infested with fake profiles.  | 54 comments](https://www.linkedin.com/posts/activity-6693809402966745088-j80Z/)\
*⚠️ Be careful out there: LinkedIn is infested with fake profiles. Out of thousands of invites I've received, hundreds have been fake. How can you recognize... 54 comments on LinkedIn*\
*www.linkedin.com*
#

## FS1264: InvoiceNet

[naiveHobo/InvoiceNet](https://github.com/naiveHobo/InvoiceNet)
*Deep neural network to extract intelligent information from invoice documents. - naiveHobo/InvoiceNet*\
*github.com*\
Do some ground work on this library and show a proof.
#

## FS1265: AWS Invoice Analyzer

Analyze AWS Invoices and recommend us 
#

## FS1266: Robinhood Graph Clone

[Sirius XM (SIRI) — Buy and sell commission-free on Robinhood](https://robinhood.com/stocks/SIRI)\
*You can buy and sell Sirius XM (SIRI) and other stocks, ETFs, and options commission-free on Robinhood with real-time quotes, market data, and relevant news. Other Robinhood Financial fees may apply, check rbnhd.co/fees for details.*\
*robinhood.com*

Clone this graph with any javascript library.
#

## FS1267: Red/Green Flags

[Disclosure Library | Robinhood](https://robinhood.com/us/en/about/legal/)\
*Commission-free investing, plus the tools you need to put your money in motion. Sign up and get your first stock for free. Certain limitations and fees may apply. View Robinhood Financial’s fee schedule at rbnhd.co/fees to learn more.*\
*robinhood.com*

Analyze these documents by using ML algorithms and show me only red/green flags. Highlight the important things as points not paragraphs.
#

## FS1268: MyLingo

Keep a lingo like urban dictionary but only for you and your friends circle. Keep Login to keep it private and share with your friends.
#

## FS1269: Episode Finder

Based on the sentence you provide, we need to find the episode. You can use script page like this https://subslikescript.com/series/Two_and_a_Half_Men-369179 and then get the episode.
#

## FS1270: WikiTable Collector

Collect table from any wiki page
#

## FS1271: Tablepedia

Wikipedia for tables alone. Show any content in table format
#

## FS1272: AndriyMLMeter

https://www.linkedin.com/in/andriyburkov/
www.linkedin.com
Out of Andriy's total posts, show how much % of ML contents are there. Show by week, month, year etc.
#

## FS1273: Data Synchronization

Bank Employees' role update from HR is not updating the Booking place. 

This issue has to be resolved

Provider Info: 
Role: Software Developer in one of the major banks in Canada 
Location: Toronto
#

## FS1274: Loss Reporting

Bank Loss, Recovery reporting is very slow it is hard to get the historical data. 

This issue has to be resolved

Provider Info: 
Role: Software Developer in one of the major banks in Canada 
Location: Toronto
#

## FS1275: Chatbot for Banking Business People

Chatbot for backend business people which is related to tools, access related, software related

Provider Info: 
Role: Software Developer in one of the major banks in Canada 
Location: Toronto
#

## FS1276: BlurMaker
<center><img src=blur.png alt="alt text" width="600" height="whatever"></center>

I need to blur the responses  by  ML
#

## FS1277: Learning Challenge Crowd Engine scoring system

* Gmail/Github Login should be used
* Without logging, i shoud be able to give score
* IP should be stored in the DB
* All entries should be dumped from Excel/CSV to MongoDB

**Tables:**

**Learning_Challenge_Entries:**

* Name
* EntryLink (unique)
* Content
* Added_at

**LCEntry_Score:**

* LCE_id
* Scorer_ip (unique)
* Scorer_email (unique)
* Score
* Added_at

**Sample entries:**

* ​https://www.linkedin.com/posts/charliecsr15_50daysofcode-50dayslearningchallenge-learningchallenge-activity-6770402296141611009-Ve8P​

* ​https://www.linkedin.com/posts/charliecsr15_50daysofcode-50dayslearningchallenge-learningchallenge-activity-6770019621832675328-QFG2​

* ​https://www.linkedin.com/posts/charliecsr15_50daysofcode-50dayslearningchallenge-learningchallenge-activity-6768886539129892864-2ssy​

* ​https://www.linkedin.com/posts/sharmila-s22_50dayslearningchallenge-learning-featurepreneur-activity-6770386881910845440-YvqP​

* https://www.linkedin.com/posts/sharmila-s22_50dayslearningchallenge-learning-featurepreneur-activity-6770007484909395968-Nfa5​

* ​https://www.linkedin.com/posts/sharmila-s22_50dayslearningchallenge-learning-featurepreneur-activity-6769653959620669440-LcdA​

* ​https://www.linkedin.com/posts/sharmila-s22_50dayslearningchallenge-day1-featurepreneur-activity-6769282703654309888-AVan​

**Phase 2:**

Keep a separate table for Name and connect with entry table
#

## Enjaami Distance
[![Enjaami Distance](https://img.youtube.com/vi/eYq7WapuDLU/hqdefault.jpg)](https://www.youtube.com/watch?v=eYq7WapuDLU "Enjaami Distance")

Find similar TRAP (Tamil Rap) songs and measure them with distance
#

## Gas Index

TBD
#

## Quotemaker

[Online Quote Maker And Generator](https://quotescover.com/tools/online-quotes-maker)\
*With this tool you can create or generate beautiful quote images with nice looking typography easily and fast.*\
*quotescover.com*\
\
Create a quote maker like this.

