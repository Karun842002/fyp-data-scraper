const puppeteer = require('puppeteer')
const cheerio = require('cheerio')
const ObjectsToCsv = require('objects-to-csv');

const baseUrl =
  'https://www.nytimes.com/search?dropmab=false&endDate=20230227&query=russia%20ukraine%20war&sort=newest&startDate=20220227'


function delay(time) {
  return new Promise(function (resolve) {
    setTimeout(resolve, time)
  })
}

const main = async () => {
  const browser = await puppeteer.launch({ headless: false })
  const page = await browser.newPage()

  await page.goto(baseUrl, { waitUntil: 'networkidle2' })

  const articleClassName = '#site-content > div.css-1wa7u5r > div:nth-child(2) > div.css-46b038'
  await page.waitForSelector(articleClassName)
  let bodyHandle = await page.$('#site-content > div.css-1wa7u5r > div:nth-child(2) > div.css-46b038')
  let html = await page.evaluate((body) => body.innerHTML, bodyHandle)
  let $ = cheerio.load(html)

  await delay(1000)
  for (let i = 0; ; ++i) {
    try {
      console.log(
        $(
          `#site-content > div.css-1wa7u5r > div:nth-child(2) > div.css-46b038 > ol > li`,
        ).length,
      )
      const button = await page.$(
        '#site-content > div.css-1wa7u5r > div:nth-child(2) > div.css-1t62hi8 > div > button',
      )
      if (!button) break
      await button.evaluate((butt) => butt.click())
      await delay(1000)
      // await page.waitForResponse((res) => {
      // 	return res.status() === 200
      // })
      bodyHandle = await page.$('body')
      html = await page.evaluate((body) => body.innerHTML, bodyHandle)
      $ = cheerio.load(html)
      let totalArticles = $(
        `#site-content > div.css-1wa7u5r > div:nth-child(2) > div.css-46b038 > ol > li`,
      ).length
      if (totalArticles > 1122) break
    }
    catch (err) { console.log(err); break; }
  }

  const articles = []
  $(
    `#site-content > div.css-1wa7u5r > div:nth-child(2) > div.css-46b038 > ol > li`,
  ).each(function (i, elem) {
    let claim = $(`#site-content > div.css-1wa7u5r > div:nth-child(2) > div.css-46b038 > ol > li:nth-child(${i + 1}) > div > div > div > a > h4`)
    let desc = $(`#site-content > div.css-1wa7u5r > div:nth-child(2) > div.css-46b038 > ol > li:nth-child(${i + 1}) > div > div > div > a > p.css-16nhkrn`)
    articles.push({ id: i, claim: claim.text(), desc: desc.text(), truth_value: 'meter-true' })
  })

  const csv = new ObjectsToCsv(articles);

  // Save to file:
  await csv.toDisk('./nyt1.csv');
  await browser.close()
}

main()
