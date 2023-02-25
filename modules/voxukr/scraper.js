const puppeteer = require('puppeteer')
const cheerio = require('cheerio')
const ObjectsToCsv = require('objects-to-csv');

const baseUrl =
	'https://voxukraine.org/en/category/voxukraine-informs/'

function delay(time) {
	return new Promise(function (resolve) {
		setTimeout(resolve, time)
	})
}

const main = async () => {
	const browser = await puppeteer.launch({ headless: false })
	const page = await browser.newPage()

	await page.goto(baseUrl, { waitUntil: 'networkidle2' })

	// const loadMoreClassName = 'body > main > section.base-section.posts-widget > div > div > div:nth-child(5) > button'
	const results = []
	for (let _ = 0; _ < 100; _++) {
		try {
			// await page.waitForSelector(loadMoreClassName)
			await delay(2000)
			let bodyHandle = await page.$('body')
			let html = await page.evaluate((body) => body.innerHTML, bodyHandle)
			let $ = cheerio.load(html)

			const news = $(
				`body > main > section.base-section.posts-widget > div > div > div:nth-child(${_ + 1})`,
			).children()

			for (let i = 0; i < news.length; ++i) {
				const claim = $(`body > main > section.base-section.posts-widget > div > div > div.posts-wrapper.d-flex.flex-column.flex-md-row.justify-content-between.justify-content-lg-start.flex-md-wrap > article:nth-child(${i + 1}) > div.post-info__content > h2`).text()
				const dateOfPublish = $(`body > main > section.base-section.posts-widget > div > div > div.posts-wrapper.d-flex.flex-column.flex-md-row.justify-content-between.justify-content-lg-start.flex-md-wrap > article:nth-child(${i + 1}) > div.post-info__content > div.post-info__date`).text()
				results.push({ claim, dateOfPublish })
			}

			const nextButton = await page.$(`body > main > section.base-section.posts-widget > div > div > div:nth-child(${_ + 2}) > button`)
			await nextButton.evaluate((butt) => butt.click())
		} catch (err) { console.log(err); break; }
	}

	const csv = new ObjectsToCsv(results);

	// Save to file:
	await csv.toDisk('./reuters.csv');
	await browser.close()
}

main()
