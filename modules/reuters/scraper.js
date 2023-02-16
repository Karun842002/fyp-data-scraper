const puppeteer = require('puppeteer')
const cheerio = require('cheerio')
const ObjectsToCsv = require('objects-to-csv');

const baseUrl =
	'https://www.reuters.com/site-search/?query=armed+conflict&offset=0&section=world&sort=newest'


function delay(time) {
	return new Promise(function (resolve) {
		setTimeout(resolve, time)
	})
}

const main = async () => {
	const browser = await puppeteer.launch({ headless: false })
	const page = await browser.newPage()

	await page.goto(baseUrl, { waitUntil: 'networkidle2' })

	const articleClassName = '.search-results__list__2SxSK'
	const results = []
	for (let _ = 0; ; _++) {
		await page.waitForSelector(articleClassName)
		let bodyHandle = await page.$('body')
		let html = await page.evaluate((body) => body.innerHTML, bodyHandle)
		let $ = cheerio.load(html)

		const news = $(
			`#fusion-app > div > div.search-layout__body__1FDkI > div.search-layout__main__L267c > div > div:nth-child(3) > div.search-results__sectionContainer__34n_c > ul > li`,
		)
		for (let i = 0; i < news.length; ++i) {
			const claim = $(`#fusion-app > div > div.search-layout__body__1FDkI > div.search-layout__main__L267c > div > div:nth-child(3) > div.search-results__sectionContainer__34n_c > ul > li:nth-child(${i + 1}) > div > div.media-story-card__body__3tRWy > a`).text()
			const dateOfPublish = $(`#fusion-app > div > div.search-layout__body__1FDkI > div.search-layout__main__L267c > div > div:nth-child(3) > div.search-results__sectionContainer__34n_c > ul > li:nth-child(${i + 1}) > div > div.media-story-card__body__3tRWy > time`).text()
			results.push({ claim, dateOfPublish })
		}

		const pageCount = $(`#fusion-app > div > div.search-layout__body__1FDkI > div.search-layout__main__L267c > div > div:nth-child(3) > div.search-results__pagination__2h60k > span`).text()
		const browsedPages = parseInt(pageCount.split('of')[0].split('to')[1])
		const totalPages = parseInt(pageCount.split('of')[1])

		if (browsedPages == totalPages) break;
		const nextButton = await page.$(`#fusion-app > div > div.search-layout__body__1FDkI > div.search-layout__main__L267c > div > div:nth-child(3) > div.search-results__pagination__2h60k > button:nth-child(3)`)
		await nextButton.evaluate((butt) => butt.click())
	}

	const csv = new ObjectsToCsv(results);

	// Save to file:
	await csv.toDisk('./reuters.csv');
	await browser.close()
}

main()
